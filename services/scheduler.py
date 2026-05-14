"""Background scheduler for daily analysis and workout completion detection."""

import asyncio
import json
from datetime import date, datetime, timedelta

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

# Rastreia URLs do LiveTrack já iniciadas para não criar sessões duplicadas
_started_livetrack_urls: set[str] = set()
# Limite de duração de uma sessão LiveTrack (3 horas)
_LIVETRACK_MAX_DURATION_HOURS = 3


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
    """Roda a cada 30min — detecta treinos concluídos RECENTEMENTE e envia review."""
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

        # Só considera treinos com dados reais (concluídos)
        completed = [w for w in all_w if any(
            w.get(f) not in (None, 0) for f in ["distance_actual", "duration_actual", "tss_actual"]
        )]

        if not completed:
            print("🔍 Nenhum treino concluído hoje.")
            return

        # Proteção: só processa treinos cuja conclusão foi recente (últimas 6h)
        # Isso evita re-processar treinos antigos após um redeploy
        cutoff = datetime.now() - timedelta(hours=6)

        for workout in completed:
            workout_id = workout.get("id")
            if not workout_id:
                continue

            # Verifica se já foi processado
            if is_workout_processed(workout_id):
                continue

            # Verifica se o treino foi completado recentemente (via end_time ou last_modified)
            completed_at_str = workout.get("completedAt") or workout.get("lastModifiedDate") or ""
            if completed_at_str:
                try:
                    # Remove milissegundos/timezone simplificado
                    completed_at = datetime.fromisoformat(completed_at_str[:19])
                    if completed_at < cutoff:
                        print(f"⏭️ Treino {workout.get('title')} ignorado — concluído há mais de 6h")
                        mark_workout_processed(workout_id)  # marca para não tentar de novo
                        continue
                except Exception:
                    pass  # sem timestamp válido, processa mesmo assim

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
    """Roda a cada 5min — detecta email do Garmin LiveTrack e inicia monitoramento com coach do TP."""
    from services.email_watcher import find_latest_livetrack_url
    from services.livetrack import parse_livetrack_url, LiveTrackSession, get_current, set_current
    from services.workout_coach import parse_workout_into_blocks, WorkoutCoach
    from tp_mcp.tools.workouts import tp_get_workouts

    current = get_current()

    # Auto-stop: encerra sessão que passou do limite de duração
    if current and current.status == "active" and current.start_time:
        elapsed_h = (datetime.now() - current.start_time).total_seconds() / 3600
        if elapsed_h >= _LIVETRACK_MAX_DURATION_HOURS:
            print(f"[Scheduler] LiveTrack auto-stop: {elapsed_h:.1f}h de atividade")
            current.stop()
            current = None

    # Nao inicia novo se ja tem sessao ativa
    if current and current.status == "active":
        return

    url = find_latest_livetrack_url()
    if not url:
        return

    # Evita iniciar sessão duplicada com a mesma URL
    if url in _started_livetrack_urls:
        return

    try:
        session_id, token = parse_livetrack_url(url)
        print(f"[Scheduler] LiveTrack detectado via email: {session_id}")

        # Para sessao anterior se existir
        if current:
            current.stop()

        # Tenta carregar treino de corrida de hoje do TP
        coach = None
        RUN_SPORTS = {"run", "running", "corrida"}
        try:
            today = date.today().isoformat()
            workouts_raw = await tp_get_workouts(today, today)
            workouts = workouts_raw.get("workouts", [])
            if workouts:
                w = next(
                    (x for x in workouts if (x.get("sport") or "").lower() in RUN_SPORTS),
                    None,
                )
                if w:
                    parsed = parse_workout_into_blocks(
                        title=w.get("title", "Treino"),
                        description=w.get("description", ""),
                        duration_planned_h=w.get("duration_planned") or 1.0,
                        tss_planned=w.get("tss_planned"),
                    )
                    if parsed and parsed.get("blocks"):
                        coach = WorkoutCoach(parsed)
                        print(f"[Scheduler] Coach estruturado: {parsed.get('nome')}")
                else:
                    print("[Scheduler] Nenhum treino de corrida hoje — LiveTrack sem coach estruturado")
        except Exception as e:
            print(f"[Scheduler] Treino do TP nao carregado: {e}")

        session = LiveTrackSession(session_id, token, workout_coach=coach)
        set_current(session)
        session.start()
        _started_livetrack_urls.add(url)
        print(f"[Scheduler] Sessao iniciada {'com coach' if coach else 'modo livre'}")
    except Exception as e:
        print(f"[Scheduler] Erro ao iniciar LiveTrack: {e}")


def start_scheduler():
    """EMERGENCY MODE — todos os jobs desabilitados para conter consumo de API."""
    # DESABILITADO: run_daily_analysis (Claude Opus, 1x/dia às 6h)
    # DESABILITADO: check_completed_workouts (Claude Sonnet, a cada 30min)
    # DESABILITADO: check_livetrack_email (IMAP Gmail, a cada 5min)
    scheduler.start()
    print("⚠️  Scheduler em modo emergencial — todos os jobs suspensos.")
