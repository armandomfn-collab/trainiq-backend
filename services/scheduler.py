"""Background scheduler for daily analysis and workout completion detection."""

import asyncio
import json
from datetime import date, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.database import (
    get_all_tokens,
    is_workout_processed,
    mark_workout_processed,
)
from services.push import send_push_notification

scheduler = AsyncIOScheduler()


async def run_daily_analysis():
    """Roda às 6h — analisa dados e envia notificação matinal."""
    print("⏰ Rodando análise diária...")
    try:
        from tp_mcp.tools.metrics import tp_get_metrics
        from tp_mcp.tools.workouts import tp_get_workouts
        from tp_mcp.tools.fitness import tp_get_fitness
        from services.ai import analyze_athlete_data

        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()

        metrics_raw = await tp_get_metrics(week_ago, today)
        workouts_raw = await tp_get_workouts(today, today)
        fitness_raw = await tp_get_fitness()

        # Extrair métricas simples
        metrics = {}
        for detail in metrics_raw.get("metrics", [{}])[-1].get("details", []):
            metrics[detail.get("label", "")] = detail.get("value")

        fitness = fitness_raw.get("current", {})
        workouts = workouts_raw.get("workouts", [])

        analysis = analyze_athlete_data(
            today=today,
            metrics=metrics,
            workouts=workouts,
            fitness=fitness,
        )

        nivel = analysis.get("nivel_prontidao", "moderado")
        status = analysis.get("status_geral", "")
        horario = analysis.get("horario_ideal", {}).get("janela", "")
        tsb = fitness.get("tsb", 0)
        hrv = metrics.get("HRV", 0)

        emoji = "🟢" if nivel == "alto" else "🟡" if nivel == "moderado" else "🔴"

        await send_push_notification(
            title=f"{emoji} Análise do dia — {date.today().strftime('%d/%m')}",
            body=f"HRV {int(hrv)} · TSB {int(tsb)} · Melhor horário: {horario}",
            data={"type": "daily_analysis", "analysis": json.dumps(analysis)},
        )
        print("✅ Análise diária enviada!")

    except Exception as e:
        print(f"❌ Erro na análise diária: {e}")


async def check_completed_workouts():
    """Roda a cada 30min — detecta treinos concluídos e envia review."""
    print("🔍 Verificando treinos concluídos...")
    try:
        from tp_mcp.tools.workouts import tp_get_workouts
        from tp_mcp.tools.metrics import tp_get_metrics
        from tp_mcp.tools.fitness import tp_get_fitness
        from services.workout_review import generate_workout_review

        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()

        workouts_raw = await tp_get_workouts(today, today)
        all_w = workouts_raw.get("workouts", [])
        completed = [w for w in all_w if w.get("type") == "completed" or any(
            w.get(f) not in (None, 0) for f in ["distance_actual", "duration_actual", "tss_actual"]
        )]

        for workout in completed:
            workout_id = workout.get("id")
            if not workout_id or is_workout_processed(workout_id):
                continue

            print(f"📊 Gerando review para: {workout.get('title')}")

            metrics_raw = await tp_get_metrics(week_ago, today)
            fitness_raw = await tp_get_fitness()

            metrics = {}
            for detail in metrics_raw.get("metrics", [{}])[-1].get("details", []):
                metrics[detail.get("label", "")] = detail.get("value")

            fitness = fitness_raw.get("current", {})

            review = generate_workout_review(
                workout=workout,
                metrics=metrics,
                fitness=fitness,
            )

            await send_push_notification(
                title=review.get("titulo_notificacao", "✅ Treino concluído!"),
                body=review.get("resumo_notificacao", "Toque para ver o review completo"),
                data={"type": "workout_review", "workout_id": workout_id, "review": json.dumps(review)},
            )

            mark_workout_processed(workout_id)
            print(f"✅ Review enviado para: {workout.get('title')}")

    except Exception as e:
        print(f"❌ Erro ao verificar treinos: {e}")


async def check_livetrack_email():
    """Roda a cada 2min — detecta email do Garmin LiveTrack e inicia monitoramento com coach do TP."""
    from services.email_watcher import find_latest_livetrack_url
    from services.livetrack import parse_livetrack_url, LiveTrackSession, get_current, set_current
    from services.workout_coach import parse_workout_into_blocks, WorkoutCoach
    from tp_mcp.tools.workouts import tp_get_workouts

    # Nao inicia novo se ja tem sessao ativa
    current = get_current()
    if current and current.status == "active":
        return

    url = find_latest_livetrack_url()
    if not url:
        return

    try:
        session_id, token = parse_livetrack_url(url)
        print(f"[Scheduler] LiveTrack detectado via email: {session_id}")

        # Para sessao anterior se existir
        if current:
            current.stop()

        # Tenta carregar treino de hoje do TP
        coach = None
        try:
            today = date.today().isoformat()
            workouts_raw = await tp_get_workouts(today, today)
            workouts = workouts_raw.get("workouts", [])
            if workouts:
                parsed = parse_workout_into_blocks(workouts[0])
                if parsed and parsed.get("blocos"):
                    coach = WorkoutCoach(parsed)
                    print(f"[Scheduler] Coach estruturado: {parsed.get('nome_treino')}")
        except Exception as e:
            print(f"[Scheduler] Treino do TP nao carregado: {e}")

        session = LiveTrackSession(session_id, token, workout_coach=coach)
        set_current(session)
        session.start()
        print(f"[Scheduler] Sessao iniciada {'com coach' if coach else 'modo livre'}")
    except Exception as e:
        print(f"[Scheduler] Erro ao iniciar LiveTrack: {e}")


def start_scheduler():
    """Inicializa o scheduler com os jobs."""
    # Análise diária às 6h da manhã
    scheduler.add_job(
        run_daily_analysis,
        CronTrigger(hour=6, minute=0),
        id="daily_analysis",
        replace_existing=True,
    )

    # Verificação de treinos concluídos a cada 30min
    scheduler.add_job(
        check_completed_workouts,
        IntervalTrigger(minutes=30),
        id="check_workouts",
        replace_existing=True,
    )

    # Monitoramento de email do Garmin LiveTrack a cada 2min
    scheduler.add_job(
        check_livetrack_email,
        IntervalTrigger(minutes=2),
        id="livetrack_email",
        replace_existing=True,
    )

    scheduler.start()
    print("Scheduler iniciado - analise as 6h, treinos a cada 30min, LiveTrack email a cada 2min")
