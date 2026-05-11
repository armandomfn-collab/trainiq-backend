"""TrainIQ Backend — FastAPI server."""

import asyncio
import os
import sys
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(override=True)

from tp_mcp.tools.fitness import tp_get_fitness
from tp_mcp.tools.metrics import tp_get_metrics
from tp_mcp.tools.workouts import tp_get_workouts, tp_update_workout

from services.ai import analyze_athlete_data
from services.push import send_push_notification
from services.database import (
    init_db, save_device_token,
    save_body_measurement, get_body_measurements, delete_body_measurement,
    save_chat_message, get_chat_history, clear_chat_history,
)
from services.body_analysis import analyze_body_composition
from services.body_vision import extract_bioimpedance_from_image
from services.scheduler import start_scheduler
from services.workout_review import generate_workout_review
from services.daily_feedback import generate_daily_feedback
from services.schedule_gen import adapt_schedule
from services.chat import chat_with_coach
from services.livetrack import parse_livetrack_url, LiveTrackSession, get_current, set_current
from services.workout_coach import parse_workout_into_blocks, WorkoutCoach

app = FastAPI(title="TrainIQ API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    start_scheduler()
    print("TrainIQ backend iniciado")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_metrics_summary(metrics_data: dict) -> dict:
    result: dict[str, Any] = {}
    metrics = metrics_data.get("metrics", [])
    if not metrics:
        return result
    today_data = metrics[-1] if metrics else {}
    for detail in today_data.get("details", []):
        label = detail.get("label", "")
        value = detail.get("value")
        if label and value is not None:
            result[label] = value
    return result


def _extract_fitness_summary(fitness_data: dict) -> dict:
    return fitness_data.get("current", {})


# ──────────────────────────────────────────────
# Endpoints principais
# ──────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "TrainIQ", "version": "0.2.0"}


@app.get("/api/dashboard")
async def get_dashboard():
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    metrics_raw, workouts_raw, fitness_raw = await asyncio.gather(
        tp_get_metrics(week_ago, today),
        tp_get_workouts(today, today),
        tp_get_fitness(),
    )

    return {
        "date": today,
        "metrics": _extract_metrics_summary(metrics_raw),
        "fitness": _extract_fitness_summary(fitness_raw),
        "workouts": workouts_raw.get("workouts", []),
    }


@app.get("/api/analysis")
async def get_analysis():
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    metrics_raw, workouts_raw, fitness_raw = await asyncio.gather(
        tp_get_metrics(week_ago, today),
        tp_get_workouts(today, today),
        tp_get_fitness(),
    )

    metrics = _extract_metrics_summary(metrics_raw)
    fitness = _extract_fitness_summary(fitness_raw)
    workouts = workouts_raw.get("workouts", [])

    analysis = analyze_athlete_data(
        today=today,
        metrics=metrics,
        workouts=workouts,
        fitness=fitness,
    )

    return {"date": today, "analysis": analysis, "workouts": workouts}


class WorkoutBlock(BaseModel):
    workout_id: str
    title: str
    sport: str | None = None
    duration_min: int
    tss_planned: float | None = None


class ScheduleRequest(BaseModel):
    date: str
    start_time: str = "05:30"
    workout_order: list[WorkoutBlock]


@app.get("/api/schedule/workouts")
async def get_schedule_workouts(date: str | None = None):
    """Retorna os treinos de uma data para montar o cronograma."""
    from datetime import date as dt
    target = date or (dt.today() + timedelta(days=1)).isoformat()
    workouts_raw = await tp_get_workouts(target, target)
    workouts = workouts_raw.get("workouts", [])

    # Monta blocos com duracao em minutos
    blocks = []
    for w in workouts:
        duration_min = int((w.get("duration_planned") or 0) * 60)
        if duration_min == 0:
            duration_min = 60  # default
        blocks.append({
            "workout_id": w["id"],
            "title": w["title"],
            "sport": w.get("sport"),
            "duration_min": duration_min,
            "tss_planned": w.get("tss_planned"),
            "description": w.get("description", ""),
        })

    return {"date": target, "blocks": blocks}


@app.post("/api/schedule/confirm")
async def confirm_schedule(req: ScheduleRequest):
    """Recebe o cronograma confirmado e retorna recomendacoes adaptadas."""
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    metrics_raw, fitness_raw = await asyncio.gather(
        tp_get_metrics(week_ago, today),
        tp_get_fitness(),
    )

    metrics = _extract_metrics_summary(metrics_raw)
    fitness = _extract_fitness_summary(fitness_raw)

    # Calcula horarios automaticamente
    from datetime import datetime, timedelta as td
    current_time = datetime.strptime(req.start_time, "%H:%M")
    workout_order = []
    for block in req.workout_order:
        workout_order.append({
            "workout_id": block.workout_id,
            "title": block.title,
            "sport": block.sport,
            "horario": current_time.strftime("%H:%M"),
            "duracao_min": block.duration_min,
            "tss_planned": block.tss_planned,
        })
        current_time += td(minutes=block.duration_min + 15)  # +15min transicao

    result = adapt_schedule(
        workout_order=workout_order,
        start_time=req.start_time,
        metrics=metrics,
        fitness=fitness,
        date=req.date,
    )

    return {"date": req.date, "schedule": result, "workout_order": workout_order}


class CreateWorkoutRequest(BaseModel):
    date: str
    title: str
    sport: str
    duration_min: int
    tss_planned: float | None = None
    description: str | None = None
    start_time: str | None = None  # "HH:MM"


@app.post("/api/workouts/create")
async def create_workout(req: CreateWorkoutRequest):
    """Cria um treino no TrainingPeaks."""
    from tp_mcp.tools.workouts import tp_create_workout
    desc = req.description or f"Treino criado pelo TrainIQ — {req.sport} {req.duration_min}min"
    result = await tp_create_workout(
        date_str=req.date,
        sport=req.sport,
        title=req.title,
        duration_minutes=req.duration_min,
        tss_planned=req.tss_planned,
        description=desc,
    )
    if result.get("isError"):
        raise HTTPException(status_code=400, detail=result.get("message", "Erro ao criar treino"))
    return {"success": True, "result": result}


@app.get("/api/daily-summary")
async def get_daily_summary():
    """Retorna status dos treinos do dia + feedback IA + preview de amanha."""
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    all_raw, tomorrow_raw, metrics_raw, fitness_raw = await asyncio.gather(
        tp_get_workouts(today, today),
        tp_get_workouts(tomorrow, tomorrow),
        tp_get_metrics(week_ago, today),
        tp_get_fitness(),
    )

    all_workouts = all_raw.get("workouts", [])
    tomorrow_workouts = tomorrow_raw.get("workouts", [])
    metrics = _extract_metrics_summary(metrics_raw)
    fitness = _extract_fitness_summary(fitness_raw)

    def _is_completed(w: dict) -> bool:
        """Detecta se um treino foi concluido pelos dados reais presentes."""
        if w.get("type") == "completed":
            return True
        # Tem dados reais registrados
        for field in ["distance_actual", "duration_actual", "tss_actual", "calories_actual"]:
            if w.get(field) not in (None, 0):
                return True
        return False

    workout_status = []
    for w in all_workouts:
        workout_status.append({**w, "completed": _is_completed(w)})

    all_done = len(workout_status) > 0 and all(w["completed"] for w in workout_status)
    any_done = any(w["completed"] for w in workout_status)

    feedback = generate_daily_feedback(
        workout_status=workout_status,
        all_done=all_done,
        tomorrow_workouts=tomorrow_workouts,
        metrics=metrics,
        fitness=fitness,
    )

    return {
        "date": today,
        "workout_status": workout_status,
        "all_done": all_done,
        "any_done": any_done,
        "tomorrow_workouts": tomorrow_workouts,
        "feedback": feedback,
    }


@app.get("/api/workout-review/{workout_id}")
async def get_workout_review(workout_id: str):
    """Retorna o review de um treino concluído."""
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    workouts_raw, metrics_raw, fitness_raw = await asyncio.gather(
        tp_get_workouts(today, today, type="completed"),
        tp_get_metrics(week_ago, today),
        tp_get_fitness(),
    )

    workouts = workouts_raw.get("workouts", [])
    workout = next((w for w in workouts if w["id"] == workout_id), None)

    if not workout:
        raise HTTPException(status_code=404, detail="Treino não encontrado ou não concluído")

    metrics = _extract_metrics_summary(metrics_raw)
    fitness = _extract_fitness_summary(fitness_raw)

    review = generate_workout_review(
        workout=workout,
        metrics=metrics,
        fitness=fitness,
    )

    return {"workout_id": workout_id, "review": review}


# ──────────────────────────────────────────────
# Push Notifications
# ──────────────────────────────────────────────

class RegisterTokenRequest(BaseModel):
    token: str


@app.post("/api/register-device")
async def register_device(req: RegisterTokenRequest):
    """Registra o token de push do dispositivo."""
    save_device_token(req.token)
    return {"success": True}


@app.post("/api/trigger-daily-analysis")
async def trigger_daily_analysis():
    """Dispara a análise diária manualmente (para testes)."""
    from services.scheduler import run_daily_analysis
    await run_daily_analysis()
    return {"success": True}


@app.post("/api/trigger-workout-check")
async def trigger_workout_check():
    """Dispara verificação de treinos concluídos manualmente."""
    from services.scheduler import check_completed_workouts
    await check_completed_workouts()
    return {"success": True}


# ──────────────────────────────────────────────
# Workouts
# ──────────────────────────────────────────────

class ApplyAdjustmentRequest(BaseModel):
    workout_id: str
    title: str | None = None
    description: str | None = None
    tss_planned: float | None = None
    coach_comment: str | None = None


@app.post("/api/workouts/apply")
async def apply_workout_adjustment(req: ApplyAdjustmentRequest):
    result = await tp_update_workout(
        workout_id=req.workout_id,
        title=req.title,
        description=req.description,
        tss_planned=req.tss_planned,
        coach_comment=req.coach_comment,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Erro ao aplicar ajuste"))
    return {"success": True, "workout_id": req.workout_id}


# ──────────────────────────────────────────────
# Body composition
# ──────────────────────────────────────────────

class BodyMeasurementRequest(BaseModel):
    date: str
    weight_kg: float
    body_fat_pct: float | None = None
    muscle_mass_kg: float | None = None
    visceral_fat: int | None = None
    water_pct: float | None = None
    bone_mass_kg: float | None = None
    bmr_kcal: int | None = None
    bmi: float | None = None
    notes: str | None = None


@app.post("/api/body/measure")
async def add_body_measure(req: BodyMeasurementRequest):
    """Salva uma medicao de bioimpedancia."""
    measure_id = save_body_measurement(req.model_dump())
    return {"success": True, "id": measure_id}


@app.get("/api/body/history")
async def get_body_history(days: int = 180):
    """Retorna historico de medicoes."""
    measurements = get_body_measurements(days)
    return {"measurements": measurements, "count": len(measurements)}


@app.delete("/api/body/measure/{measure_id}")
async def remove_body_measure(measure_id: int):
    """Remove uma medicao."""
    ok = delete_body_measurement(measure_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Medicao nao encontrada")
    return {"success": True}


class ExtractImageRequest(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"


@app.post("/api/body/extract-image")
async def extract_body_image(req: ExtractImageRequest):
    """Extrai valores de bioimpedancia de uma foto usando Claude Vision."""
    try:
        result = await extract_bioimpedance_from_image(req.image_base64, req.media_type)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Nao foi possivel extrair os valores da imagem: {str(e)}")


@app.get("/api/body/analysis")
async def get_body_analysis():
    """Analise IA cruzando composicao corporal com treinos."""
    measurements = get_body_measurements(days=180)
    if not measurements:
        return {"analysis": None, "message": "Registre ao menos uma medicao para receber analise."}

    four_weeks_ago = (date.today() - timedelta(days=28)).isoformat()
    workouts_raw, fitness_raw = await asyncio.gather(
        tp_get_workouts(four_weeks_ago, date.today().isoformat()),
        tp_get_fitness(),
    )
    workouts  = workouts_raw.get("workouts", [])
    fitness   = _extract_fitness_summary(fitness_raw)
    analysis  = analyze_body_composition(measurements, workouts, fitness)
    return {"analysis": analysis, "latest": measurements[0], "count": len(measurements)}


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────

class ChatMessageModel(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str  # só a nova mensagem do usuário


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat com o coach IA com histórico persistente e contexto do TrainingPeaks."""
    today     = date.today().isoformat()
    week_ago  = (date.today() - timedelta(days=7)).isoformat()
    tomorrow  = (date.today() + timedelta(days=1)).isoformat()

    try:
        metrics_raw, workouts_raw, fitness_raw, tomorrow_raw = await asyncio.gather(
            tp_get_metrics(week_ago, today),
            tp_get_workouts(today, today),
            tp_get_fitness(),
            tp_get_workouts(tomorrow, tomorrow),
        )
        context = {
            "data_hoje": today,
            "treinos_hoje": workouts_raw.get("workouts", []),
            "treinos_amanha": tomorrow_raw.get("workouts", []),
            "metricas": _extract_metrics_summary(metrics_raw),
            "forma": _extract_fitness_summary(fitness_raw),
        }
    except Exception:
        context = {"data_hoje": today}

    # Salva mensagem do usuário
    save_chat_message("user", req.message)

    # Monta histórico completo para o Claude (últimas 40 msgs)
    messages = get_chat_history(limit=40)

    reply = await chat_with_coach(messages, context)

    # Salva resposta do coach
    save_chat_message("assistant", reply)

    return {"reply": reply, "history": get_chat_history(limit=40)}


@app.get("/api/chat/history")
async def chat_history():
    """Retorna o histórico completo do chat."""
    return {"history": get_chat_history(limit=60)}


@app.delete("/api/chat/history")
async def chat_history_clear():
    """Limpa o histórico do chat."""
    clear_chat_history()
    return {"success": True}


# ──────────────────────────────────────────────
# LiveTrack
# ──────────────────────────────────────────────

@app.get("/api/livetrack/workout-preview")
async def livetrack_workout_preview(date: str | None = None):
    """Busca o treino do TP para a data e retorna blocos estruturados para o coach."""
    import traceback
    from datetime import date as dt
    target = date or dt.today().isoformat()

    try:
        workouts_raw = await tp_get_workouts(target, target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar treinos do TP: {e}\n{traceback.format_exc()}")

    workouts = workouts_raw.get("workouts", [])

    if not workouts:
        return {"workout": None, "blocks": [], "message": "Nenhum treino encontrado para essa data."}

    # Pega o primeiro treino do dia (ou o de corrida se houver)
    workout = next((w for w in workouts if (w.get("sport") or "").lower() in ("run", "running", "corrida")), workouts[0])

    try:
        parsed = parse_workout_into_blocks(
            title=workout.get("title", "Treino"),
            description=workout.get("description", ""),
            duration_planned_h=workout.get("duration_planned") or 1.0,
            tss_planned=workout.get("tss_planned"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao parsear blocos: {e}\n{traceback.format_exc()}")

    blocks_out = [
        {
            "index": b.index, "name": b.name,
            "duration_min": b.duration_min, "zone": b.zone,
            "target_pace_min": b.target_pace_min,
            "target_pace_max": b.target_pace_max,
            "description": b.description, "tip": b.tip,
        }
        for b in parsed["blocks"]
    ]

    return {
        "workout_id":    workout.get("id"),
        "workout_title": parsed["nome"],
        "abertura":      parsed["abertura"],
        "total_min":     parsed["total_min"],
        "blocks":        blocks_out,
    }


class LiveTrackStartRequest(BaseModel):
    url: str
    workout_id: str | None = None   # se fornecido, ativa o coach estruturado
    date: str | None = None          # data do treino (default: hoje)


@app.post("/api/livetrack/start")
async def livetrack_start(req: LiveTrackStartRequest):
    """Inicia monitoramento de uma sessao LiveTrack do Garmin."""
    current = get_current()
    if current and current.status in ("active", "waiting"):
        current.stop()

    try:
        session_id, token = parse_livetrack_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── Cria coach estruturado se date ou workout_id fornecidos ──────────
    coach = None
    coach_info = {}
    if req.date or req.workout_id:
        try:
            from datetime import date as dt
            target = req.date or dt.today().isoformat()
            workouts_raw = await tp_get_workouts(target, target)
            workouts = workouts_raw.get("workouts", [])

            if workouts:
                # Prefere corrida; senão pega o primeiro
                workout = next(
                    (w for w in workouts if w.get("sport", "").lower() in ("run", "running", "corrida")),
                    workouts[0]
                )
                # Se workout_id fornecido, tenta encontrar o workout exato
                if req.workout_id:
                    exact = next((w for w in workouts if str(w.get("id")) == str(req.workout_id)), None)
                    if exact:
                        workout = exact

                parsed = parse_workout_into_blocks(
                    title=workout.get("title", "Treino"),
                    description=workout.get("description", ""),
                    duration_planned_h=workout.get("duration_planned") or 1.0,
                    tss_planned=workout.get("tss_planned"),
                )
                coach = WorkoutCoach(parsed)
                coach_info = {
                    "workout_title": parsed["nome"],
                    "total_min": parsed["total_min"],
                    "blocks_count": len(parsed["blocks"]),
                }
                print(f"[LiveTrack] Coach estruturado: {parsed['nome']} — {len(parsed['blocks'])} blocos")
        except Exception as e:
            print(f"[LiveTrack] Erro ao criar coach estruturado: {e}")
            # Continua em modo livre

    session = LiveTrackSession(session_id, token, workout_coach=coach)
    set_current(session)
    session.start()
    return {
        "success": True,
        "session_id": session_id,
        "status": "waiting",
        "coach_mode": coach is not None,
        **coach_info,
    }


@app.get("/api/livetrack/status")
async def livetrack_status():
    """Retorna status atual da sessao LiveTrack e o ultimo alerta de coaching."""
    current = get_current()
    if not current:
        return {"status": "none", "current_metrics": {}, "latest_alert": None, "alert_time": None}
    return current.to_dict()


class LiveTrackAlertRequest(BaseModel):
    text: str

@app.post("/api/livetrack/test-alert")
async def livetrack_test_alert(req: LiveTrackAlertRequest):
    """Injeta um alerta manualmente na sessao atual (para testes)."""
    from datetime import datetime
    current = get_current()
    if not current:
        raise HTTPException(status_code=404, detail="Nenhuma sessao ativa")
    current.latest_alert = req.text
    current.alert_time   = datetime.now()
    return {"success": True, "alert": req.text}

@app.post("/api/livetrack/stop")
async def livetrack_stop():
    """Encerra o monitoramento."""
    current = get_current()
    if current:
        current.stop()
        set_current(None)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
