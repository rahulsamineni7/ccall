"""
routes/calls.py — Call management endpoints.

Handles: listing phone numbers, making outbound calls, fetching call history,
hanging up calls, and processing Twilio status webhooks.
"""
import os
import json
import uuid
import asyncio
import urllib.parse
import logging

from fastapi import APIRouter, Request, Depends, HTTPException, Response, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List

from dependencies import get_current_user
from twilio_provisioning import get_provisioner
from backend.supabase_client import supabase

logger = logging.getLogger("alora.calls")

router = APIRouter(prefix="/api/calls", tags=["Calls"])

# ── In-memory campaign status (lightweight, no extra DB table needed) ────────
CAMPAIGN_STATUS = {}  # campaign_id -> {total, completed, failed, in_progress, results: []}

# ── Pydantic Models ──────────────────────────────────────────────────────────

class MakeCallRequest(BaseModel):
    to: str
    from_: str = Field(..., alias="from")
    context: Optional[dict] = {}
    system_prompt: Optional[str] = None


class Contact(BaseModel):
    phone_number: str
    context: Optional[dict] = {}


class BulkCallRequest(BaseModel):
    from_number: str = Field(..., alias="from")
    contacts: List[Contact]
    system_prompt: Optional[str] = None

# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/numbers")
async def list_twilio_numbers(user: dict = Depends(get_current_user)):
    """List all Twilio phone numbers available on the main account."""
    try:
        provisioner = get_provisioner()
        client = provisioner.client

        incoming_numbers = client.incoming_phone_numbers.list(limit=50)

        numbers = []
        for record in incoming_numbers:
            numbers.append({
                "phone_number": record.phone_number,
                "friendly_name": record.friendly_name,
                "sid": record.sid
            })

        return {"numbers": numbers}
    except Exception as e:
        logger.error(f"Error fetching numbers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch phone numbers: {str(e)}")


@router.post("/make")
async def make_outbound_call(call_req: MakeCallRequest, request: Request, user: dict = Depends(get_current_user)):
    """Initiate an outbound call using Twilio with context."""
    try:
        provisioner = get_provisioner()
        client = provisioner.client

        logger.info(f"Making call from {call_req.from_} to {call_req.to}")

        # Dynamic Host for TwiML
        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        if not base_url:
            host = request.headers.get("host", "localhost")
            proto = "https" if "localhost" not in host else "http"
            base_url = f"{proto}://{host}"

        ctx = call_req.context or {}
        ctx['call_type'] = 'outbound'

        # --- WORKFLOW ROUTING (OUTBOUND) ---
        system_prompt = call_req.system_prompt

        try:
            from backend.supabase_client import supabase_adapter
            wf = supabase_adapter.get_active_workflow_for_trigger(user.id, trigger_type='call_ended', run_on='outbound')
            if wf:
                logger.info(f"Linked Outbound Workflow: {wf['name']}")
                ctx['workflow_id'] = wf['id']
        except Exception as e:
            logger.warning(f"Workflow lookup error: {e}")

        context_json = json.dumps(ctx)
        context_encoded = urllib.parse.quote(context_json)
        prompt_encoded = urllib.parse.quote(system_prompt or "")

        # Sanitize phone number
        clean_to = call_req.to.replace(" ", "").replace("-", "")
        phone_encoded = urllib.parse.quote(clean_to)

        twiml_url = f"{base_url}/twilio/incoming?call_type=outbound&phone={phone_encoded}&context={context_encoded}&prompt={prompt_encoded}"

        call = client.calls.create(
            to=call_req.to,
            from_=call_req.from_,
            url=twiml_url,
            status_callback=f"{base_url}/api/calls/status",
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            status_callback_method='POST'
        )

        # Persist to Database
        try:
            supabase.table("outbound_calls").insert({
                "user_id": user.id,
                "call_sid": call.sid,
                "to_number": call_req.to,
                "from_number": call_req.from_,
                "status": call.status,
                "system_instruction": call_req.system_prompt,
                "context_data": call_req.context,
                "active_rules": []
            }).execute()
        except Exception as db_err:
            logger.error(f"DB Insert Error: {db_err}")

        logger.info(f"Call created: {call.sid}, Status: {call.status}")

        return {
            "call_sid": call.sid,
            "status": call.status,
            "to": call_req.to,
            "from": call_req.from_
        }
    except Exception as e:
        logger.error(f"Error making call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make call: {str(e)}")


@router.get("/active")
async def get_active_calls(user: dict = Depends(get_current_user)):
    """Get all currently active/in-progress calls."""
    try:
        provisioner = get_provisioner()
        client = provisioner.client

        active_statuses = ['queued', 'ringing', 'in-progress']
        calls = []

        for status in active_statuses:
            active_calls = client.calls.list(status=status, limit=20)
            for call in active_calls:
                calls.append({
                    "sid": call.sid,
                    "to": call.to,
                    "from": call.from_,
                    "status": call.status,
                    "duration": call.duration,
                    "start_time": str(call.start_time) if call.start_time else None
                })

        return {"calls": calls}
    except Exception as e:
        logger.error(f"Error fetching active calls: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch active calls: {str(e)}")


@router.get("/history")
async def get_call_history(
    page: int = 0,
    limit: int = 20,
    user: dict = Depends(get_current_user)
):
    """Get call history with pagination."""
    try:
        start = page * limit
        end = start + limit - 1

        res = supabase.table("outbound_calls")\
            .select("*")\
            .eq("user_id", user.id)\
            .order("created_at", desc=True)\
            .range(start, end)\
            .execute()

        calls = res.data if res.data else []

        formatted_calls = []
        for c in calls:
            formatted_calls.append({
                "sid": c['call_sid'],
                "to": c['to_number'],
                "from": c['from_number'],
                "status": c['status'],
                "duration": c['duration'],
                "price": c['cost'],
                "start_time": c['created_at'],
                "date_created": c['created_at'],
                "system_prompt": c.get('system_prompt')
            })

        return {"calls": formatted_calls, "page": page}
    except Exception as e:
        logger.error(f"Error fetching call history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch call history: {str(e)}")


@router.post("/{call_sid}/hangup")
async def hangup_call(call_sid: str, user: dict = Depends(get_current_user)):
    """End/hangup an active call."""
    try:
        provisioner = get_provisioner()
        client = provisioner.client

        logger.info(f"Hanging up call: {call_sid}")
        call = client.calls(call_sid).update(status='completed')

        return {
            "call_sid": call.sid,
            "status": call.status
        }
    except Exception as e:
        logger.error(f"Error hanging up call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to hangup call: {str(e)}")


@router.post("/status")
async def call_status_webhook(request: Request):
    """Webhook to receive call status updates from Twilio."""
    try:
        data = await request.form()
        call_sid = data.get('CallSid')
        status = data.get('CallStatus')
        duration = data.get('CallDuration')

        update_data = {"status": status}
        if duration:
            update_data["duration"] = int(duration)

        if call_sid:
            supabase.table("outbound_calls").update(update_data).eq("call_sid", call_sid).execute()

    except Exception as e:
        logger.error(f"Error processing status webhook: {e}")

    return Response(content="<Response/>", media_type="application/xml")


# ── Bulk Calling ─────────────────────────────────────────────────────────────

async def _process_bulk_campaign(
    campaign_id: str,
    user: dict,
    from_number: str,
    contacts: list[Contact],
    system_prompt: str | None,
    base_url: str,
):
    """Background worker: dispatches calls one-by-one with rate limiting."""
    provisioner = get_provisioner()
    client = provisioner.client

    CAMPAIGN_STATUS[campaign_id] = {
        "total": len(contacts),
        "completed": 0,
        "failed": 0,
        "in_progress": 0,
        "results": [],
    }
    status = CAMPAIGN_STATUS[campaign_id]

    for i, contact in enumerate(contacts):
        clean_to = contact.phone_number.replace(" ", "").replace("-", "")
        ctx = contact.context or {}
        ctx["call_type"] = "outbound"
        ctx["campaign_id"] = campaign_id
        ctx["campaign_index"] = i

        context_json = json.dumps(ctx)
        context_encoded = urllib.parse.quote(context_json)
        prompt_encoded = urllib.parse.quote(system_prompt or "")
        phone_encoded = urllib.parse.quote(clean_to)

        twiml_url = (
            f"{base_url}/twilio/incoming?call_type=outbound"
            f"&phone={phone_encoded}&context={context_encoded}&prompt={prompt_encoded}"
        )

        try:
            call = client.calls.create(
                to=clean_to,
                from_=from_number,
                url=twiml_url,
                status_callback=f"{base_url}/api/calls/status",
                status_callback_event=["initiated", "ringing", "answered", "completed"],
                status_callback_method="POST",
            )

            # Persist to DB
            try:
                supabase.table("outbound_calls").insert({
                    "user_id": user.id,
                    "call_sid": call.sid,
                    "to_number": clean_to,
                    "from_number": from_number,
                    "status": call.status,
                    "system_instruction": system_prompt,
                    "context_data": ctx,
                    "active_rules": [],
                }).execute()
            except Exception as db_err:
                logger.error(f"Bulk call DB insert error: {db_err}")

            status["completed"] += 1
            status["results"].append({
                "phone": clean_to,
                "call_sid": call.sid,
                "status": call.status,
            })
            logger.info(f"Campaign {campaign_id} [{i+1}/{len(contacts)}] → {clean_to}: {call.status}")

        except Exception as e:
            status["failed"] += 1
            status["results"].append({
                "phone": clean_to,
                "call_sid": None,
                "status": "failed",
                "error": str(e),
            })
            logger.error(f"Campaign {campaign_id} [{i+1}/{len(contacts)}] → {clean_to}: FAILED - {e}")

        # ── Rate limit: 1 call per second (Twilio standard CPS) ──
        if i < len(contacts) - 1:
            await asyncio.sleep(1)

    logger.info(f"Campaign {campaign_id} finished: {status['completed']} ok, {status['failed']} failed")


@router.post("/bulk")
async def bulk_outbound_call(
    bulk_req: BulkCallRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Launch a bulk outbound calling campaign. Returns immediately with a campaign_id."""
    if not bulk_req.contacts:
        raise HTTPException(status_code=400, detail="Contact list is empty")
    if len(bulk_req.contacts) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 contacts per campaign")

    campaign_id = str(uuid.uuid4())

    # Resolve base URL
    base_url = os.environ.get("BASE_URL", "").rstrip("/")
    if not base_url:
        host = request.headers.get("host", "localhost")
        proto = "https" if "localhost" not in host else "http"
        base_url = f"{proto}://{host}"

    background_tasks.add_task(
        _process_bulk_campaign,
        campaign_id=campaign_id,
        user=user,
        from_number=bulk_req.from_number,
        contacts=bulk_req.contacts,
        system_prompt=bulk_req.system_prompt,
        base_url=base_url,
    )

    logger.info(f"Bulk campaign {campaign_id} queued: {len(bulk_req.contacts)} contacts")

    return {
        "campaign_id": campaign_id,
        "status": "queued",
        "total_contacts": len(bulk_req.contacts),
    }


@router.get("/campaigns/{campaign_id}")
async def get_campaign_status(campaign_id: str, user: dict = Depends(get_current_user)):
    """Get the current status of a bulk calling campaign."""
    if campaign_id in CAMPAIGN_STATUS:
        return {"campaign_id": campaign_id, **CAMPAIGN_STATUS[campaign_id]}

    # Fallback: query DB for calls tagged with this campaign_id
    try:
        res = (
            supabase.table("outbound_calls")
            .select("call_sid, to_number, status, context_data")
            .eq("user_id", user.id)
            .execute()
        )
        matches = [
            r for r in (res.data or [])
            if isinstance(r.get("context_data"), dict)
            and r["context_data"].get("campaign_id") == campaign_id
        ]
        if matches:
            completed = sum(1 for m in matches if m["status"] == "completed")
            failed = sum(1 for m in matches if m["status"] == "failed")
            return {
                "campaign_id": campaign_id,
                "total": len(matches),
                "completed": completed,
                "failed": failed,
                "in_progress": len(matches) - completed - failed,
                "results": [
                    {"phone": m["to_number"], "call_sid": m["call_sid"], "status": m["status"]}
                    for m in matches
                ],
            }
    except Exception as e:
        logger.error(f"Campaign lookup error: {e}")

    raise HTTPException(status_code=404, detail="Campaign not found")

