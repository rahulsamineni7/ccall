import os
import copy
import re
import time
import asyncio
import uuid
import json
import csv
import io
from datetime import datetime, timedelta
from groq import AsyncGroq
from cartesia import AsyncCartesia
from backend.supabase_client import supabase_adapter



MANDATORY_INSTRUCTIONS = "\n\n[CRITICAL]: When your conversation objective is met (e.g., booking confirmed, question answered, or you have collected the required information) or the user says goodbye, you MUST call the `end_call` tool IMMEDIATELY in that same turn. Do not say goodbye and wait for a response. The call must be cut immediately after your final information is provided."

class GroqAgent:
    def __init__(self, api_key, phone=None, output_folder=".", output_sample_rate=24000, automation_rules=None, user_token=None, enable_calendar=False, user_settings=None, call_context=None):
        self.client = AsyncGroq(api_key=api_key) # Async Client
        self.phone = phone
        self.output_folder = output_folder
        self.output_sample_rate = output_sample_rate
        self.user_token = user_token # Store User OAuth Token for Automation resolution
        
        # Init DB Adapter (Supabase)
        self.db = supabase_adapter
        
        # Load Config: Prefer Injected Settings, Fallback to Empty
        if user_settings:
            self.config = user_settings
        else:
            # Fallback (mostly for local testing without server injection)
            self.config = {}
        
        # Cartesia API key: prefer .env, fallback to user config
        cartesia_key = os.environ.get("CARTESIA_API_KEY") or self.config.get("cartesia_api_key")
        if not cartesia_key:
            print("[WARN] No Cartesia API Key found. TTS will be unavailable.")

        self.cartesia = AsyncCartesia(api_key=cartesia_key) if cartesia_key else None
        self.voice_id = self.config.get("voice_id")
        if not self.voice_id or len(self.voice_id) < 10 or "debug" in self.voice_id or "cartesia_uuid" in self.voice_id:
            self.voice_id = "bec003e2-3cb3-429c-8468-206a393c67ad"
        self.system_instruction = self.config.get("system_instruction") or "You are a helpful AI assistant."
        
        self.automation_rules = automation_rules or []
        


        # CALENDAR TOGGLE LOGIC
        if not enable_calendar:
            self.system_instruction += "\n\n[POLICY]: Calendar Booking is currently DISABLED. If the user asks to book an appointment, politely apologize and say you cannot book appointments right now."
        else:
            self.system_instruction += """

[POLICY]: Calendar Booking is ENABLED. You MUST use the provided real-time calendar tools.

**REQUIRED WORKFLOW for Bookings:**
1. **TRIGGER:** User asks for an appointment, test drive, or mentions a time (e.g., "tomorrow at 5").
2. **ACTION:** Call `check_calendar_availability` IMMEDIATELY with the inferred date and time. Do NOT ask, just check.
3. **DECISION:**
   - If AVAILABLE: Ask for Name/Phone (if missing), then call `book_test_drive`.
   - If BUSY: Suggest the next available slot.
   
**EXAMPLES:**
User: "Can I come tomorrow at 5?"
You: (Call Tool: check_calendar_availability(date='tomorrow', time='5 PM'))
...Function returns available...
You: "That time is open! May I have your name to book it?"

User: "Book for Rahul at 10am tomorrow"
You: (Call Tool: check_calendar_availability(date='tomorrow', time='10 AM'))
...Function returns available...
You: (Call Tool: book_test_drive(customer_name='Rahul', ...))
"Blocked for Rahul!"

**DO NOT** just discuss times - ACTUALLY CHECK and BOOK using the tools.
"""

        # KNOWLEDGE BASE INJECTION (CSV/Manual)
        knowledge_items = self.config.get("knowledge_base", [])
        if knowledge_items:
            kb_text = "\n\n[KNOWLEDGE BASE]:\n"
            for item in knowledge_items:
                if item.get("is_active", True):
                    kb_text += f"- {item.get('title')}: {item.get('content')}\n"
            self.system_instruction += kb_text

        # NOTE: Automation Rules are NO LONGER injected into the Conversational Prompt.
        
        # Append mandatory call termination instructions (cannot be overwritten by custom prompts)
        self._append_mandatory_instructions()

        self.model_stt = "whisper-large-v3-turbo"
        self.model_llm = "llama-3.3-70b-versatile"
        self.history = []
        
        # Google OAuth tokens: prefer config (WebSocket path) over user_token param (Twilio path)
        self.google_tokens = self.config.get('google_tokens') or self.user_token
        self.enable_calendar = enable_calendar
        
        # Define function tools for Groq API
        self.tools = []
        
        # 1. Calendar Tools (Conditional)
        if enable_calendar:
            self.tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "check_calendar_availability",
                        "description": "Check if a specific date/time is available.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string", "description": "Date (YYYY-MM-DD or 'tomorrow')"},
                                "time": {"type": "string", "description": "Time (e.g. '3:00 PM')"},
                                "duration_minutes": {"type": "integer", "description": "Duration (default 60)"}
                            },
                            "required": ["date", "time"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "book_test_drive",
                        "description": "Book a confirmed test drive appointment.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "time": {"type": "string"},
                                "customer_name": {"type": "string"},
                                "phone_number": {"type": "string"},
                                "car_model": {"type": "string"},
                                "notes": {"type": "string"}
                            },
                            "required": ["date", "time", "customer_name", "phone_number", "car_model"]
                        }
                    }
                },
                 {
                    "type": "function",
                    "function": {
                        "name": "list_bookings",
                        "description": "List upcoming bookings.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "max_results": {"type": "integer"}
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_booking",
                        "description": "Update an existing booking.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "booking_date": {"type": "string"},
                                "booking_time": {"type": "string"},
                                "new_date": {"type": "string"},
                                "new_time": {"type": "string"},
                                "new_car_model": {"type": "string"},
                                "customer_name": {"type": "string"}
                            },
                            "required": ["booking_date", "booking_time"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_booking",
                        "description": "Cancel an existing booking.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "booking_date": {"type": "string"},
                                "booking_time": {"type": "string"},
                                "customer_name": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["booking_date", "booking_time"]
                        }
                    }
                }
            ])

        # 2. Core Tools (Always Available)
        self.tools.append({
            "type": "function",
            "function": {
                "name": "end_call",
                "description": "End the call immediately. Use this when the conversation objective is met or user says goodbye.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for ending the call (optional)"}
                    },
                    "required": []
                }
            }
        })
        
        # DEBUG: Log tool initialization
        print(f"[INIT] Calendar enabled param: {enable_calendar}")
        print(f"[INIT] Tools defined: {len(self.tools)} tools")
        
        # Call Context Injection
        if call_context:
            try:
                # Format context nicely
                context_str = json.dumps(call_context, indent=2)
                self.system_instruction += f"\n\n[CALL CONTEXT / CUSTOMER DETAILS]:\nUse the following details to guide your conversation with the user. Treat this as known information about the person you are calling:\n{context_str}\n"
                print(f"[INIT] Injected Call Context: {list(call_context.keys())}")
            except Exception as e:
                print(f"[INIT] Error injecting call context: {e}")
        if len(self.tools) > 0:
            for tool in self.tools:
                print(f"[INIT] - Available tool: {tool['function']['name']}")

    def _append_mandatory_instructions(self):
        """Ensures critical termination instructions are always present."""
        if MANDATORY_INSTRUCTIONS not in self.system_instruction:
            self.system_instruction += MANDATORY_INSTRUCTIONS
                
    def update_call_context(self, call_context):
        """
        Update the system instruction with call-specific context (from TwiML parameters).
        """
        if call_context:
            try:
                context_str = json.dumps(call_context, indent=2)
                self.system_instruction += f"\n\n[CALL CONTEXT / CUSTOMER DETAILS]:\nUse the following details to guide your conversation with the user. Treat this as known information about the person you are calling:\n{context_str}\n"
                self._append_mandatory_instructions()
                print(f"[AGENT] Updated system prompt with context: {list(call_context.keys())}")
            except Exception as e:
                print(f"[AGENT] Error updating call context: {e}")
        
    def update_system_prompt(self, prompt):
        """
        Update the system instruction with custom user prompt.
        """
        if prompt:
            # Overwrite the system instruction with the specific prompt
            self.system_instruction = prompt
            self._append_mandatory_instructions()
            print(f"[AGENT] Overwrote system prompt with custom instructions: {prompt[:50]}...")
        
    def _get_voice_settings(self, text):
        """
        Determines the best Voice ID, Language Code, and Model ID based on text.
        """
        # Regex for Scripts
        if re.search(r'[\u0900-\u097F]', text): # Devanagari (Hindi)
            return self.voice_id, "hi", "sonic-multilingual"
        if re.search(r'[\u0C00-\u0C7F]', text): # Telugu
            return self.voice_id, "te", "sonic-3"
        if re.search(r'[\u0B80-\u0BFF]', text): # Tamil
            return self.voice_id, "ta", "sonic-3"
        if re.search(r'[\u0C80-\u0CFF]', text): # Kannada
            return self.voice_id, "kn", "sonic-3"
        if re.search(r'[\u0D00-\u0D7F]', text): # Malayalam
            return self.voice_id, "ml", "sonic-3"
        if re.search(r'[\u0600-\u06FF]', text): # Urdu
            return self.voice_id, "ur", "sonic-multilingual"
            
        # Default to English (Indian Accent - Arushi)
        return self.voice_id, "en", "sonic-english"

    def _is_hallucination(self, text):
        """
        Check for common Whisper hallucinations on silence/noise.
        """
        text = text.strip()
        if not text: return True
        if len(text) < 2: return True # Ignore single chars "w", "a"
        
        # Common Whisper Hallucinations
        hallucinations = [
            "Uncertain.", "MBC news.", "Copyright", "Amara.org",
            "Good morning, Mr Jane.", "Visualstudio.com"
        ]
        if text in hallucinations: return True
        if text.lower() in ["thank you", "thanks", "hello", "bye", "ok", "okay"]:
            # Context Check: Previously checked for noise, now allowing valid greetings.
            pass
            
        return False

    async def generate_greeting(self):
        """
        Generates an initial greeting based on the System Prompt.
        """
        try:
            # DEBUG: Print system prompt at call start
            print(f"\n{'='*60}\n[INIT] SYSTEM PROMPT FOR THIS CALL:\n{'='*60}")
            print(self.system_instruction)
            print(f"{'='*60}\n")
            
            # Construct Prompt
            messages = [{"role": "system", "content": self.system_instruction}]
            messages.append({"role": "user", "content": "(System: Call started. Greet the user.)"})

            
            completion = await self.client.chat.completions.create(
                model=self.model_llm,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            text = completion.choices[0].message.content
            print(f"[AGENT] Generated Greeting: {text}")
            
            # Update History
            self.history.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            print(f"[ERROR] Greeting Generation Failed: {e}")
            return "Hello." # Fallback
        
    async def process_audio_stream(self, audio_bytes):
        """
        Full Async Streaming Pipeline: STT -> LLM (Stream) -> TTS (Stream)
        """
        t_start = time.time()
        
        # 1. Reload Config (Disabled to preserve Injected User Settings)
        # self.config = self.db.get_config()
        
        # Load system instruction ONLY from database (app_options table)
        # DISABLE RELOAD to preserve Dynamic Prompts (Outbound Calls)
        # self.system_instruction = self.config.get("system_instruction", "")
        
        # Add knowledge base if available
        # kb_text = self.db.get_knowledge_base()
        # if kb_text: self.system_instruction += f"\n\n### KNOWLEDGE BASE ###\n{kb_text}"
        
        # Add customer context if phone number available
        # if self.phone:
        #    customer = self.db.get_customer(self.phone)
        #    if customer:
        #        self.system_instruction += f"\n\n### CUSTOMER ###\nName: {customer.get('Name')}\nHistory: {customer.get('History')}"
        
        # Add inventory
        # inventory = self.db.get_inventory()
        # if inventory: self.system_instruction += f"\n\nINVENTORY:\n{inventory}"

        try:
            # 2. STT (Async)
            print("[INFO] Whisper Listening...")
            file_obj = io.BytesIO(audio_bytes)
            file_obj.name = "audio.webm"
            
            # Map codes into full names for better Prompt Biasing
            code_map = {"hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu", "en": "English"}
            supported_codes = self.config.get("supported_languages", ["en"])
            supported_names = [code_map.get(c, c) for c in supported_codes]
            
            # Stronger Prompt with Native Script Hints
            # Providing a "fake" previous context in the prompt helps bias the model.
            prompt_str = (
                f"The user may speak in {', '.join(supported_names)}. "
                "Transcribe Hindi in Devanagari (e.g. नमस्ते), "
                "Telugu in Telugu Script (e.g. నమస్కారం), "
                "Tamil in Tamil Script (e.g. வணக்கம்). "
                "Do not translate. Transcribe exactly what is spoken."
            )

            # Async Whisper Call
            t_stt_start = time.time()
            transcription = await self.client.audio.transcriptions.create(
                file=("audio.webm", file_obj),
                model=self.model_stt,
                prompt=prompt_str,
                response_format="verbose_json",
                temperature=0.0 # Deterministic
            )
            print(f"[TIME] STT Duration: {time.time() - t_stt_start:.3f}s")
            
            user_text = transcription.text.strip()
            detected_language = getattr(transcription, 'language', 'unknown')
            print(f"[STT] Detected: '{detected_language}' | Text: {user_text}")
            
            # --- LANGUAGE FILTER ---
            # (Keep existing filter logic or move to helper)
            # For brevity/safety, let's keep basic check here or trust Native/Prompt.
            # Native API usually handles this well.
            
            # --- HALLUCINATION FILTER ---
            # Whisper often hallucinates on silence. Filter known garbage.
            if self._is_hallucination(user_text):
                print(f"[STT] Ignored Hallucination: '{user_text}'")
                return

            # Delegate to shared generator with explicit language hint
            async for data, audio in self._generate_response(user_text, lang_code_hint=detected_language):
                yield data, audio

        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            yield {"agent": f"Error: {e}"}, None

    async def process_text_stream(self, user_text):
        """
        Input: Text (from Web Speech API).
        output: Stream (same format as process_audio_stream).
        """
        if not user_text or len(user_text) < 1: return
        
        # Log Language Detection
        _, lang_code, _ = self._get_voice_settings(user_text)
        print(f"[INFO] Processing Text Input: {user_text}")
        print(f"[LANG] Detected Text Language: {lang_code}")
        
        async for data, audio in self._generate_response(user_text):
            yield data, audio

    async def _generate_response(self, user_text, lang_code_hint=None):
        """
        Shared Logic: User Text -> History -> LLM Stream -> TTS Stream
        """
        if not user_text: return

        print(f"[DEBUG] _generate_response called with: '{user_text[:50]}...'")
        print(f"[DEBUG] self.tools length: {len(self.tools)}")

        # 1. Update History
        yield {"user": user_text}, None
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 10: self.history = self.history[-10:]
        
        # --- HEURISTIC: END CALL DETECTION ---
        # If user says "bye" or "goodbye", we force an end call signal.
        if re.search(r"\b(bye|goodbye|see you|hang up|talk to you later)\b", user_text, re.IGNORECASE):
            print(f"[HEURISTIC] End Call Keyword detected: {user_text}")
            # Yield final goodbye (optional, or let LLM say it via tool?)
            # Usually better to let LLM say it if we aren't forcing hard stop.
            # But to guarantee stop, we can set a flag or force a tool.
            # For now, let's inject a system instruction override for this turn?
            # Or just check it after response? 
            # Let's simple return a canned goodbye and the end signal to be fast.
            
            # Simple Canned Goodbye (multilingual safe-ish, or use English default)
            yield {"agent": "Goodbye!"}, None
            # We can generate audio for "Goodbye" using TTS if needed, but let's just Close.
            yield {"control": "end_call"}, None
            return

        t_start = time.time()
        
        # Determine Language
        # 1. Use STT Hint if available (High Confidence)
        # 2. Else, use Regex on text (Fallback)
        voice_id_unused, regex_lang, _ = self._get_voice_settings(user_text)
        
        lang_code = lang_code_hint if lang_code_hint else regex_lang
        
        print(f"[DECISION] Lang Source: {'STT' if lang_code_hint else 'Regex'} | Code: {lang_code}")
        
        # Dynamic System Instruction
        dynamic_sys = self.system_instruction
        
        # DEBUG: Print the actual system prompt being sent
        print(f"\n{'='*60}\n[DEBUG] SYSTEM PROMPT BEING SENT TO LLM:\n{'='*60}")
        print(dynamic_sys)
        print(f"{'='*60}\n")
        
        # FORCE Language Mirroring (With Bilingual/Code-Switching Support)
        # FORCE Language Mirroring (With Bilingual/Code-Switching Support)
        # Language detected for logging/TTS selection, but NO hardcoded instructions injected.
        if lang_code in ['hi', 'hindi', 'te', 'telugu', 'ta', 'tamil', 'kn', 'kannada', 'ml', 'malayalam', 'ur', 'urdu']:
            # Optional: You can keep a minimal hint if needed, OR remove entirely as requested.
            # For now, relying strictly on DB system instruction as requested.
            pass

        # 2. LLM Generation
        # --- FUNCTION TOOL CALLING (Calendar Operations) ---
        # Use tools if calendar is enabled and defined
        run_tools = len(self.tools) > 0
        
        # DEBUG: Log tool status
        if run_tools:
            print(f"[DEBUG] Tools available: {len(self.tools)} tools defined")
            for tool in self.tools:
                print(f"[DEBUG] - Tool: {tool['function']['name']}")
        else:
            print(f"[DEBUG] No tools available (calendar disabled or not configured)")
        
        # If tools enabled, check if LLM wants to call them
        tool_outputs = None
        should_end_call = False

        if run_tools:
            print(f"[THINK] Checking Tools... (Lang: {lang_code})")
            # Inject Date for reference
            import datetime
            date_context = f"\nToday is: {datetime.date.today()}. ISO Format: {datetime.datetime.now().isoformat()}"
            
            try:
                check_response = await self.client.chat.completions.create(
                    model=self.model_llm,
                    messages=[{"role": "system", "content": dynamic_sys + date_context}] + self.history,
                    tools=self.tools,
                    tool_choice="auto",
                    stream=False
                )
                msg = check_response.choices[0].message
            except Exception as tool_error:
                # Handle function calling errors (e.g., invalid syntax)
                print(f"[TOOL] Function call failed: {tool_error}")
                print("[TOOL] Falling back to regular response without tools...")
                msg = None  # Set to None to skip tool execution
            
            if msg and msg.tool_calls:
                fn_name = msg.tool_calls[0].function.name
                print(f"[TOOL] Model requested: {fn_name}")

                # --- FILLER INJECTION ---
                if fn_name != "end_call":
                    # Speak "Wait..." before doing the work
                    try:
                        # Simple Filler Dictionary
                        fillers = {
                            "en": "Just a moment, checking availability...",
                            "hi": "Ek minute, main check kar raha hoon...",
                            "te": "Okka nimisham, check chestunnanu...",
                            "ta": "Oru nimidam, check panren...",
                            "kn": "Ondu nimisha, check madtini...",
                        }
                        filler_text = fillers.get(lang_code, "Just a moment, checking...")
                        
                        # Get Voice ID (Reusing logic or just using current)
                        # We need to send this to user
                        voice_id_filler, _, model_id_filler = self._get_voice_settings(filler_text)
                        
                        async with self.cartesia.tts.websocket_connect() as ws_filler:
                            ctx_filler = ws_filler.context()
                            await ctx_filler.send(
                                model_id=model_id_filler,
                                transcript=filler_text,
                                voice={"mode": "id", "id": voice_id_filler},
                                output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.output_sample_rate},
                                language=lang_code
                            )
                            # Yield Audio to Twilio immediately
                            async for event in ctx_filler.receive():
                                if event.type == "chunk" and hasattr(event, "audio") and event.audio:
                                     yield None, event.audio
                    except Exception as e:
                        print(f"[FILLER] Error: {e}")
                # ------------------------

                # Execute Tool using new handler
                fn_name = msg.tool_calls[0].function.name
                fn_args = json.loads(msg.tool_calls[0].function.arguments)
                
                print(f"[TOOL] Calling: {fn_name}")
                tool_result = await self._execute_tool(fn_name, fn_args)
                
                # Check for Hangup
                if isinstance(tool_result, dict) and tool_result.get("action") == "hangup":
                    should_end_call = True
                    print("[AGENT] End Call Tool Triggered. Will hang up after response.")

                # Add to history
                self.history.append(msg) # Tool Call
                self.history.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_calls[0].id,
                    "content": json.dumps(tool_result)
                })
                print(f"[TOOL] Result: {tool_result}")

        # 3. Final Streaming Response
        print(f"[THINK] Llama Streaming... (Lang: {lang_code})")
        stream = await self.client.chat.completions.create(
            model=self.model_llm,
            messages=[{"role": "system", "content": dynamic_sys}] + self.history,
            temperature=0.6,
            max_tokens=256,
            stream=True 
        )

        # 3. Stream Processing Buffer
        buffer = ""
        current_sentence = ""
        full_response = ""
        
        # Open TTS WebSocket with Fallback
        try:
            async with self.cartesia.tts.websocket_connect() as ws:
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    buffer += token
                    current_sentence += token
                    full_response += token
                    
                    # Heuristic: Check for sentence delimiters
                    if any(punct in token for punct in [".", "?", "!", "\n", "sir", "mam", ":"]):
                        # If sentence is long enough, send it
                        if len(current_sentence.strip()) > 5: # Avoid "Hi." latency
                                print(f"[STREAM] Sent to TTS: {current_sentence}")
                                
                                # Measure TTS Latency
                                t_tts_start = time.time()
                                first_byte = True

                                # Stream TTS audio chunks directly
                                voice_id, lang_code, model_id = self._get_voice_settings(current_sentence)
                                ctx = ws.context()
                                await ctx.send(
                                    model_id=model_id,
                                    transcript=current_sentence,
                                    voice={"mode": "id", "id": voice_id},
                                    output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.output_sample_rate},
                                    language=lang_code
                                )
                                async for event in ctx.receive():
                                    if event.type == "chunk" and hasattr(event, "audio") and event.audio:
                                        if first_byte:
                                            print(f"[TIME] TTS Latency: {time.time() - t_tts_start:.3f}s")
                                            first_byte = False
                                        yield None, event.audio
                                
                                yield {"agent": current_sentence}, None # Update UI incrementally
                                current_sentence = ""

                # Flush remaining buffer
                if current_sentence.strip():
                    print(f"[STREAM] Sent Final: {current_sentence}")
                    voice_id, lang_code, model_id = self._get_voice_settings(current_sentence)
                    ctx = ws.context()
                    await ctx.send(
                        model_id=model_id,
                        transcript=current_sentence,
                        voice={"mode": "id", "id": voice_id},
                        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
                        language=lang_code
                    )
                    async for event in ctx.receive():
                        if event.type == "chunk" and hasattr(event, "audio") and event.audio:
                            yield None, event.audio
                    yield {"agent": current_sentence}, None

        except Exception as e:
            print(f"[CARTESIA] Connection Failed: {e}.")
            yield {"error": "TTS Connection Failed"}, None
            return
            
        finally:
            # TRIGGER HANGUP IF REQUESTED
            if should_end_call:
                 print("[AGENT] Sending End Call Signal to Server...")
                 yield {"control": "end_call"}, None
        
        # Update History with Full Response
        self.history.append({"role": "assistant", "content": full_response})
        print(f"[TIME] Total Latency: {time.time() - t_start:.2f}s")
        
        # --- POST CALL ANALYSIS (Action Tag Parsing) ---
        # Regex to find [ACTION: FUNCTION(Args)]
        # Example: [ACTION: BOOK_TEST_DRIVE(Rahul, 10am)]
        action_match = re.search(r'\[ACTION:\s*(.*?)\]', full_response)
        if action_match:
            action_content = action_match.group(1).strip()
            print(f"[ANALYSIS] Detected Action: {action_content}")
            
            # Simple parsing: Split by '(' to get intent Name, then args
            # Or just log the whole string as intent
            
            # Try to extract Name if possible (heuristic)
            # Default to "Customer" if not found
            customer_name = "Customer"
            
            # Log to DB
            # We use the phone number from the session (self.phone)
            # Intent is the raw action string
            try:
                # self.db.log_lead(customer_name, self.phone, action_content)
                print(f"[ANALYSIS] Lead Logging Dispatched (Automation Engine): {action_content}")
            except Exception as e:
                print(f"[ANALYSIS] Logging Failed: {e}")
        
        # Action Check (Post-Processing)
        # self._check_for_actions(full_response)

    async def run_post_call_automation(self, transcript_lines):
        """
        Analyzes the full transcript after the call ends.
        Executes automation rules based on the entire conversation.
        """
        if not self.automation_rules or not transcript_lines:
            print("[AUTO] No rules or empty transcript. Skipping.")
            return

        print(f"[AUTO] Starting Post-Call Analysis on {len(transcript_lines)} lines...")
        
        # 1. Build Context for Analysis
        transcript_text = "\n".join(transcript_lines)
        
        prompt = f"""You are a Data Extraction Engine.
Analyze the following conversation transcript and extract information based on the configured rules.

TRANSCRIPT:
{transcript_text}

AUTOMATION RULES:"""

        for rule in self.automation_rules:
            base_instr = f"\n- Service: {rule.get('service')}, Target: {rule.get('resource_name')}. Instruction: {rule.get('instruction')}"
            
            # Re-fetch Headers Context (in case it wasn't valid at init, though unlikely to change mid-call)
            # Actually, better to fetch now or reuse if we cached it?
            # We didn't cache headers in init, we just put them in string. 
            # Let's fetch again or just trust the rule instruction if it's clear.
            # Ideally fetch headers again to be precise.
            # if rule.get('service') == 'sheets' and rule.get('resource_id'):
            #      headers = self.db.get_sheet_headers(rule.get('resource_id'), user_token=self.user_token, sheet_name=rule.get('resource_name'))
            #      if headers:
            #          base_instr += f"\n  [SCHEMA] Columns: {headers}"
            
            # Context Injection (Calendar)
            if rule.get('service') == 'calendar':
                base_instr += "\n  [SCHEMA] Fields: summary, start_time (ISO 8601), duration_minutes"

            prompt += base_instr

        prompt += """
        
        IMPLICIT RULES (Always Active):
        - If the user explicitly requests to "Book a Test Drive" or agrees to a "Test Drive" appointment with a specific date/time, YOU MUST EXTRACT IT as a 'calendar' action.
        - Use "Test Drive - {Customer Name}" as the summary.
        - Default duration: 60 minutes.

INSTRUCTIONS:
1. Return a JSON object with a key "actions".
2. "actions" should be a list of objects, each containing:
   - "service": "sheets" or "calendar"
   - "resource_id": The target ID from rules (or 'primary' for calendar)
   - "data": 
       - For Sheets: Flat JSON object mapping Column Names to Values.
       - For Calendar: {"summary": "Meeting Title", "start_time": "YYYY-MM-DDTHH:MM:SS", "duration_minutes": 60, "description": "Context"}
3. If information is missing for a column, omit it or use "N/A".
4. Do not hallucinate. Only extract what was explicitly said.
5. For Calendar: Infer the date relative to 'today' (Assume today is {datetime.date.today()}).

Example JSON Output:
{
  "actions": [
    {
      "service": "sheets",
      "resource_id": "carcustomer",
      "data": {"Name": "Rahul", "Phone": "...", "Car Model": "Nexon"}
    },
    {
      "service": "calendar",
      "resource_id": "primary",
      "data": {"summary": "Test Drive", "start_time": "2023-10-25T14:00:00"}
    }
  ]
}
Return ONLY JSON."""

        # 2. Call LLM
        try:
            print("[AUTO] Sending to LLM...")
            # Inject Date for reference
            import datetime
            prompt = prompt.replace("{datetime.date.today()}", str(datetime.date.today()))
            
            completion = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant", # Fast model sufficient for extraction
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = completion.choices[0].message.content
            print(f"[AUTO] LLM Result: {result}")
            
            # 3. Parse and Execute
            data = json.loads(result)
            actions = data.get("actions", [])
            
            for action in actions:
                if action.get("service") == "sheets":
                    await self._execute_sheet_update(action)
                elif action.get("service") == "calendar":
                    print(f"[AUTO] Booking Calendar Event: {action.get('data')}")
                    # Run sync function in thread or direct
                    # result = self.db.create_calendar_event(action.get('data'), user_token=self.user_token)
                    result = "Moved to Automation Engine"
                    if result and "FAILED_CONFLICT" in str(result):
                        print(f"[AUTO] Booking FAILED due to Conflict: {result}")
                        # Ideally, we should log this failure somewhere visible (e.g. Activity Log DB)
                    elif result:
                         print(f"[AUTO] Booking Success: {result}")
                    else:
                         print("[AUTO] Booking Failed (Unknown Error).")

        except Exception as e:
            print(f"[AUTO] Analysis Failed: {e}")

    async def _execute_sheet_update(self, action):
        resource_id = action.get("resource_id")
        data = action.get("data") # Dict {Col: Val}
        if not resource_id or not data: return
        
        print(f"[AUTO] Writing to Sheet {resource_id}: {data}")
        # Delegate to DB Adapter (Simulated Async)
        # We need to implement `append_to_sheet_smart` in DBAdapter
        # Since DBAdapter is synchronous (gspread), we run it directly.
        # self.db.append_to_sheet_smart(resource_id, data, user_token=self.user_token)
        print("[AUTO] Sheet Update Mocked (DBAdapter Removed)")

    async def say(self, text):
        """
        Generates TTS Audio for a given text (without LLM/STT).
        Useful for Greetings.
        """
        ws = None
        try:
            voice_id, lang_code, model_id = self._get_voice_settings(text)
            print(f"[INFO] Saying: {text} | Voice: {voice_id} | Lang: {lang_code} | Model: {model_id}")
            
            # 1. Yield Text Event (for logging)
            yield {"agent": text}, None

            async with self.cartesia.tts.websocket_connect() as ws:
                ctx = ws.context()
                await ctx.send(
                    model_id=model_id,
                    transcript=text,
                    voice={"mode": "id", "id": voice_id},
                    output_format={
                        "container": "raw", 
                        "encoding": "pcm_s16le", 
                        "sample_rate": self.output_sample_rate
                    },
                    language=lang_code
                )

                async for event in ctx.receive():
                    if getattr(event, "type", None) == "chunk" and hasattr(event, "audio") and event.audio:
                        yield None, event.audio

        except Exception as e:
            print(f"Cartesia Say Error: {e}")

        # Legacy Action Engine Removed
    
    async def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute agent function tools"""
        print(f"[TOOL] Executing: {tool_name} with args: {arguments}")
        
        try:
            if tool_name == "check_calendar_availability":
                return await self._check_calendar_availability(
                    arguments.get('date'),
                    arguments.get('time'),
                    arguments.get('duration_minutes', 60)
                )
            
            elif tool_name == "book_test_drive":
                return await self._book_test_drive(arguments)
            
            elif tool_name == "list_bookings":
                return await self._list_bookings(arguments.get('max_results', 5))
            
            elif tool_name == "update_booking":
                return await self._update_booking(arguments)
            
            elif tool_name == "cancel_booking":
                return await self._cancel_booking(arguments)
            
            elif tool_name == "end_call":
                 print(f"[TOOL] END CALL REQUESTED. Reason: {arguments.get('reason')}")
                 return {"action": "hangup", "status": "completing", "message": "Call ended by agent."}
            
            return {"error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            print(f"[TOOL ERROR] {tool_name}: {e}")
            return {"error": str(e)}
    
    async def _check_calendar_availability(self, date_str: str, time_str: str, duration: int = 60) -> dict:
        """Check if a calendar slot is available"""
        try:
            # Import calendar functions
            from automation_engine import read_calendar_events, check_availability, get_google_access_token
            from calendar_utils import parse_datetime, format_datetime_iso
            from datetime import timedelta, datetime
            
            if not self.google_tokens:
                print(f"[TOOL-DEBUG] self.google_tokens is: {self.google_tokens}")
                print(f"[TOOL-DEBUG] self.config keys: {list(self.config.keys()) if self.config else 'None'}")
                return {"available": False, "error": "Google Calendar not connected"}
            
            print(f"[TOOL-DEBUG] Tokens available: {list(self.google_tokens.keys())}")
            
            # Get access token
            access_token = await get_google_access_token(self.google_tokens)
            if not access_token:
                return {"available": False, "error": "Failed to authenticate with Google Calendar"}
            
            # Parse datetime
            start_dt = await parse_datetime(date_str, time_str)
            end_dt = start_dt + timedelta(minutes=duration)
            
            start_iso = format_datetime_iso(start_dt)
            end_iso = format_datetime_iso(end_dt)
            
            print(f"[TOOL] Checking availability: {start_iso} to {end_iso}")
            
            # Read existing events
            existing_events = await read_calendar_events(start_iso, end_iso, access_token)
            
            # Check availability
            availability = check_availability(start_iso, end_iso, existing_events)
            
            return {
                "available": availability['available'],
                "message": availability['message'],
                "conflicts": availability.get('conflicts', []),
                "requested_time": f"{date_str} at {time_str}"
            }
        
        except Exception as e:
            print(f"[TOOL ERROR] check_calendar_availability: {e}")
            return {"available": False, "error": str(e)}
    
    async def _book_test_drive(self, details: dict) -> dict:
        """Book a test drive appointment with conflict detection and update capability"""
        try:
            from automation_engine import (
                create_google_calendar_event, 
                update_google_calendar_event,
                get_google_access_token,
                read_calendar_events,
                check_availability
            )
            from calendar_utils import parse_datetime, format_datetime_iso
            from datetime import timedelta, datetime
            
            if not self.google_tokens:
                return {"success": False, "error": "Google Calendar not connected"}
            
            # Get access token
            access_token = await get_google_access_token(self.google_tokens)
            if not access_token:
                return {"success": False, "error": "Failed to authenticate"}
            
            # Parse datetime
            start_dt = await parse_datetime(details['date'], details['time'])
            end_dt = start_dt + timedelta(minutes=60)  # Default 1 hour
            
            start_iso = format_datetime_iso(start_dt)
            end_iso = format_datetime_iso(end_dt)
            
            print(f"[TOOL] Checking for conflicts: {start_iso} to {end_iso}")
            
            # Step 1: Check for existing bookings in this time slot
            existing_events = await read_calendar_events(start_iso, end_iso, access_token)
            availability = check_availability(start_iso, end_iso, existing_events)
            
            # Create event object in Google Calendar API format
            event = {
                "summary": f"Test Drive - {details.get('car_model', 'Luxury Car')}",
                "description": f"Customer test drive booking\nName: {details['customer_name']}\nPhone: {details.get('phone_number', 'N/A')}\nCar: {details.get('car_model', 'Not specified')}",
                "location": "Dealership Showroom",
                "start": {
                    "dateTime": start_iso,
                    "timeZone": "Asia/Kolkata"
                },
                "end": {
                    "dateTime": end_iso,
                    "timeZone": "Asia/Kolkata"
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 30}
                    ]
                }
            }
            
            # Step 2: If conflict exists, update the existing event instead of creating new
            if not availability['available'] and availability.get('conflicts'):
                conflict = availability['conflicts'][0]  # Get first conflicting event
                existing_event_id = conflict.get('event_id')
                
                if existing_event_id:
                    print(f"[TOOL] Conflict found! Updating existing event: {existing_event_id}")
                    print(f"[TOOL] Existing event: {conflict.get('title')} at {conflict.get('start')}")
                    
                    # Update the existing event with new details
                    result = await update_google_calendar_event(existing_event_id, event, access_token)
                    
                    if result.get('success'):
                        return {
                            "success": True,
                            "message": f"Updated existing booking for {details['date']} at {details['time']}",
                            "event_id": result.get('event_id'),
                            "calendar_link": result.get('link'),
                            "action": "updated"
                        }
                    else:
                        return {"success": False, "error": f"Failed to update: {result.get('error')}"}
                else:
                    # Conflict exists but no event_id (shouldn't happen), reject
                    return {
                        "success": False, 
                        "error": f"Time slot already booked: {conflict.get('title')} at {conflict.get('start')}"
                    }
            
            # Step 3: No conflict - create new event
            print(f"[TOOL] No conflicts. Creating new calendar event: {event['summary']}")
            print(f"[TOOL] Event time: {start_iso} to {end_iso}")
            
            result = await create_google_calendar_event(event, access_token)
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": f"Test drive booked for {details['date']} at {details['time']}",
                    "event_id": result.get('event_id'),
                    "calendar_link": result.get('link'),
                    "action": "created"
                }
            else:
                return {"success": False, "error": result.get('error', 'Booking failed')}
        
        except Exception as e:
            print(f"[TOOL ERROR] book_test_drive: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def _list_bookings(self, max_results: int = 5) -> dict:
        """List upcoming test drive bookings"""
        try:
            from automation_engine import get_google_access_token, get_upcoming_bookings
            
            if not self.google_tokens:
                return {"success": False, "error": "Google Calendar not connected"}
            
            access_token = await get_google_access_token(self.google_tokens)
            if not access_token:
                return {"success": False, "error": "Failed to authenticate"}
            
            bookings = await get_upcoming_bookings(access_token, max_results)
            
            if bookings:
                return {
                    "success": True,
                    "count": len(bookings),
                    "bookings": bookings,
                    "message": f"Found {len(bookings)} upcoming test drive(s)"
                }
            else:
                return {
                    "success": True,
                    "count": 0,
                    "bookings": [],
                    "message": "No upcoming test drive bookings found"
                }
                
        except Exception as e:
            print(f"[TOOL ERROR] list_bookings: {e}")
            return {"success": False, "error": str(e)}

    async def _update_booking(self, details: dict) -> dict:
        """Update an existing test drive booking"""
        try:
            from automation_engine import (
                get_google_access_token, 
                read_calendar_events,
                update_google_calendar_event
            )
            from calendar_utils import parse_datetime, format_datetime_iso
            from datetime import timedelta
            
            if not self.google_tokens:
                return {"success": False, "error": "Google Calendar not connected"}
            
            access_token = await get_google_access_token(self.google_tokens)
            if not access_token:
                return {"success": False, "error": "Failed to authenticate"}
            
            # Find the existing booking
            original_start = await parse_datetime(details['booking_date'], details['booking_time'])
            original_end = original_start + timedelta(minutes=60)
            
            start_iso = format_datetime_iso(original_start)
            end_iso = format_datetime_iso(original_end)
            
            print(f"[TOOL] Looking for booking at: {start_iso}")
            
            existing_events = await read_calendar_events(start_iso, end_iso, access_token)
            
            if not existing_events:
                return {"success": False, "error": f"No booking found at {details['booking_date']} {details['booking_time']}"}
            
            # Find test drive event
            target_event = None
            for event in existing_events:
                if "Test Drive" in event.get("summary", ""):
                    target_event = event
                    break
            
            if not target_event:
                return {"success": False, "error": "No test drive booking found at that time"}
            
            event_id = target_event.get("id")
            print(f"[TOOL] Found booking to update: {event_id}")
            
            # Build update object
            updates = {}
            
            if details.get('new_date') or details.get('new_time'):
                new_date = details.get('new_date', details['booking_date'])
                new_time = details.get('new_time', details['booking_time'])
                new_start = await parse_datetime(new_date, new_time)
                new_end = new_start + timedelta(minutes=60)
                
                updates["start"] = {"dateTime": format_datetime_iso(new_start), "timeZone": "Asia/Kolkata"}
                updates["end"] = {"dateTime": format_datetime_iso(new_end), "timeZone": "Asia/Kolkata"}
            
            if details.get('new_car_model'):
                updates["summary"] = f"Test Drive - {details['new_car_model']}"
            
            if details.get('customer_name'):
                current_desc = target_event.get("description", "")
                updates["description"] = f"Customer: {details['customer_name']}\n{current_desc}"
            
            if not updates:
                return {"success": False, "error": "No changes specified"}
            
            result = await update_google_calendar_event(event_id, updates, access_token)
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": f"Booking updated successfully",
                    "event_id": result.get('event_id'),
                    "calendar_link": result.get('link')
                }
            else:
                return {"success": False, "error": result.get('error')}
                
        except Exception as e:
            print(f"[TOOL ERROR] update_booking: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def _cancel_booking(self, details: dict) -> dict:
        """Cancel an existing test drive booking"""
        try:
            from automation_engine import (
                get_google_access_token, 
                read_calendar_events,
                delete_google_calendar_event
            )
            from calendar_utils import parse_datetime, format_datetime_iso
            from datetime import timedelta
            
            if not self.google_tokens:
                return {"success": False, "error": "Google Calendar not connected"}
            
            access_token = await get_google_access_token(self.google_tokens)
            if not access_token:
                return {"success": False, "error": "Failed to authenticate"}
            
            # Find the booking to cancel
            cancel_start = await parse_datetime(details['booking_date'], details['booking_time'])
            cancel_end = cancel_start + timedelta(minutes=60)
            
            start_iso = format_datetime_iso(cancel_start)
            end_iso = format_datetime_iso(cancel_end)
            
            print(f"[TOOL] Looking for booking to cancel at: {start_iso}")
            
            existing_events = await read_calendar_events(start_iso, end_iso, access_token)
            
            if not existing_events:
                return {"success": False, "error": f"No booking found at {details['booking_date']} {details['booking_time']}"}
            
            # Find test drive event
            target_event = None
            for event in existing_events:
                if "Test Drive" in event.get("summary", ""):
                    target_event = event
                    break
            
            if not target_event:
                return {"success": False, "error": "No test drive booking found at that time"}
            
            event_id = target_event.get("id")
            event_title = target_event.get("summary", "Test Drive")
            print(f"[TOOL] Found booking to cancel: {event_id} - {event_title}")
            
            result = await delete_google_calendar_event(event_id, access_token)
            
            if result.get('success'):
                reason = details.get('reason', 'No reason provided')
                return {
                    "success": True,
                    "message": f"Booking '{event_title}' at {details['booking_date']} {details['booking_time']} has been cancelled",
                    "reason": reason,
                    "cancelled_event_id": event_id
                }
            else:
                return {"success": False, "error": result.get('error')}
                
        except Exception as e:
            print(f"[TOOL ERROR] cancel_booking: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

