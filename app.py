# app.py
import os
import json
import uuid
import boto3
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from database import setup_database, Session, Customer, EMI, OTP, UnresolvedChat, ClientInteraction, RAGDocument
from utils import generate_otp_code, send_sms_otp, verify_otp_code, extract_digits_from_speech, hide_number
from taskrouter_setup import setup as taskrouter_config_setup, build_client as build_twilio_client, WorkspaceInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex()) # Flask session key

# Load environment variables
HOST = os.getenv("HOST")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID")
CLAUDE_INTENT_MODEL_ID = os.getenv("CLAUDE_INTENT_MODEL_ID")
ALICE_NUMBER = os.getenv("ALICE_NUMBER") # Used for TaskRouter setup script

# Initialize Bedrock client
bedrock_runtime_client = None
try:
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    logger.info("🏆 AWS Bedrock runtime client initialized successfully!")
except Exception as e:
    logger.error(f"❌ Error initializing AWS Bedrock runtime client: {e}")

# In-memory session store for voice calls (CallSid -> session_data)
# In production, use Redis or a proper session store for persistence and scale
VOICE_SESSION_STORE = {}

# Twilio TaskRouter client and workspace info
TASKROUTER_CLIENT = None
WORKSPACE_INFO: WorkspaceInfo = None

# --- Helper Functions for Ozonetel Responses ---
# IMPORTANT: These functions assume Ozonetel expects a JSON response for call control.
# YOU MUST verify Ozonetel's API documentation for the exact JSON/XML format.

def ozonetel_speak_and_listen(text):
    """Generates a JSON response for Ozonetel to speak text and then listen for user input."""
    logger.info(f"Ozonetel Response: SPEAK '{text}' and LISTEN")
    return jsonify({"speak": text, "listen": True})

def ozonetel_speak_and_hangup(text):
    """Generates a JSON response for Ozonetel to speak text and then hang up the call."""
    logger.info(f"Ozonetel Response: SPEAK '{text}' and HANGUP")
    return jsonify({"speak": text, "hangup": True})

def ozonetel_dial_agent(agent_phone_number):
    """
    Generates a JSON response for Ozonetel to dial an agent and bridge the call.
    This is a conceptual representation. Ozonetel's actual 'dial' mechanism
    might be different (e.g., specific XML or more complex JSON).
    """
    logger.info(f"Ozonetel Response: DIAL agent {agent_phone_number}")
    # Ozonetel might require a 'from' number (your Ozonetel virtual number)
    # You might need to add `callerId: os.getenv("OZONETEL_PHONE_NUMBER")` if required
    return jsonify({
        "dial": {
            "number": agent_phone_number,
            "timeout": 30, # seconds to wait for agent to answer
            "record": False # adjust as needed
        }
    })

# --- LLM Functions (Bedrock with RAG) ---

def call_bedrock_llm(prompt, model_id, system_prompt=None):
    """
    Calls the Bedrock LLM with the given prompt.
    """
    if not bedrock_runtime_client:
        logger.error("Bedrock client not initialized. Cannot make LLM call.")
        return "Sorry, the AI service is currently unavailable."

    messages = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt}) # Claude 3 uses user role for system instructions

    messages.append({"role": "user", "content": prompt})


    body = json.dumps({
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.5,
        "top_p": 0.9
    })

    try:
        response = bedrock_runtime_client.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except Exception as e:
        logger.error(f"Error invoking Bedrock LLM {model_id}: {e}", exc_info=True)
        return "I apologize, I'm having trouble processing your request right now. Please try again later."


def classify_intent_with_llm(user_input):
    """
    Classifies the user's intent using Bedrock LLM.
    Possible intents: 'query_emi', 'live_agent_request', 'unclear'.
    """
    system_prompt = """
    You are an intent classification system. Analyze the user's query to determine their primary intent.
    Possible intents are:
    - 'query_emi': The user is asking about their EMI (Equated Monthly Installment) or loan details.
    - 'live_agent_request': The user explicitly wants to talk to a human agent, connect to support, or speak with a representative.
    - 'unclear': The intent cannot be clearly determined from the query or falls outside the defined intents.

    Respond with ONLY the intent name (e.g., 'query_emi', 'live_agent_request', 'unclear').
    Do not include any other text, explanation, or punctuation.
    """
    prompt = f"User query: \"{user_input}\""
    response = call_bedrock_llm(prompt, model_id=CLAUDE_INTENT_MODEL_ID, system_prompt=system_prompt)
    response = response.strip().lower()

    if "query_emi" in response:
        return "query_emi"
    elif "live_agent_request" in response:
        return "live_agent_request"
    else:
        return "unclear"

def generate_emi_response_with_rag(emi_details, customer_info):
    """
    Generates a natural language response for EMI details using Bedrock LLM with RAG.
    """
    if not emi_details or not customer_info:
        return "I couldn't find your EMI details. Please ensure your account ID is correct."

    # Prepare EMI data for LLM
    emi_data_str = f"""
    Loan ID: {emi_details.loan_id}
    Principal Amount: {emi_details.principal_amount:.2f}
    Interest Rate: {emi_details.interest_rate:.2f}%
    Tenure (months): {emi_details.tenure_months}
    Monthly EMI Amount: {emi_details.amount_due:.2f}
    Next Due Date: {emi_details.due_date.strftime('%Y-%m-%d') if emi_details.due_date else 'N/A'}
    Next Amount Due: {emi_details.amount_due:.2f if emi_details.amount_due is not None else 'N/A'}
    Status: {emi_details.status}
    Last Payment Date: {emi_details.payment_date.strftime('%Y-%m-%d') if emi_details.payment_date else 'N/A'}
    Amount Paid (last): {emi_details.amount_paid:.2f if emi_details.amount_paid is not None else 'N/A'}
    """

    system_prompt = f"""
    You are a helpful and polite financial assistant providing EMI details.
    Here is the customer's information:
    - Customer ID: {customer_info.customer_id}
    - Account ID: {customer_info.account_id}
    - Name: {customer_info.full_name}
    - Phone Number (masked): {hide_number(customer_info.phone_number)}

    Here is the retrieved EMI data for the customer's loan:
    {emi_data_str}

    Based on the above information, provide a concise and clear breakdown of the customer's EMI.
    Start by greeting the customer by name. State their monthly EMI, the next due date, and the amount due.
    Mention if the EMI is paid or pending. Keep the response natural for a voice interaction.
    """
    prompt = "Please provide the EMI details for the customer."
    response = call_bedrock_llm(prompt, model_id=CLAUDE_MODEL_ID, system_prompt=system_prompt)
    return response

# --- Flask Routes ---

@app.route('/')
def home():
    return "Financial Voice Bot is running. Ready to receive calls!"

@app.route('/ozonetel_voice_webhook', methods=['POST'])
def handle_ozonetel_voice_call():
    call_sid = request.form.get('CallSid') # Ozonetel Call SID
    from_number = request.form.get('From') # Customer's phone number
    user_speech_result = request.form.get('SpeechResult', '').strip() # Transcribed speech from Ozonetel

    logger.info(f"Received Ozonetel webhook for CallSid: {call_sid}, From: {from_number}, Speech: '{user_speech_result}'")

    # Retrieve or initialize session data for this CallSid
    # Use uuid for session_id for ClientInteraction logging
    session_id_str = VOICE_SESSION_STORE.get(call_sid, {}).get('session_id')
    current_session_id = uuid.UUID(session_id_str) if session_id_str else uuid.uuid4()
    session_data = VOICE_SESSION_STORE.get(call_sid, {'stage': 'initial_greeting', 'intent': None, 'session_id': str(current_session_id)})


    try:
        # Log user's speech input (if not initial greeting and there's speech)
        if user_speech_result and session_data['stage'] != 'initial_greeting':
            with Session() as db_session:
                ClientInteraction(
                    session_id=current_session_id,
                    customer_id=session_data.get('customer_id', 'N/A'),
                    sender='user',
                    message_text=user_speech_result,
                    stage=session_data['stage'],
                    intent=session_data.get('intent')
                )
                db_session.add(ClientInteraction(
                    session_id=current_session_id,
                    customer_id=session_data.get('customer_id', 'N/A'),
                    sender='user',
                    message_text=user_speech_result,
                    stage=session_data['stage'],
                    intent=session_data.get('intent')
                ))
                db_session.commit()

        response_text = ""
        # Initial greeting
        if session_data['stage'] == 'initial_greeting':
            session_data['stage'] = 'awaiting_query'
            VOICE_SESSION_STORE[call_sid] = session_data
            response_text = "Hello! Welcome to our financial assistant. How can I help you today?"
            return ozonetel_speak_and_listen(response_text)

        # Awaiting user query after greeting or previous prompt
        if session_data['stage'] == 'awaiting_query':
            if not user_speech_result:
                response_text = "I didn't catch that. Please tell me your query, like 'What is my EMI?' or 'Connect to agent'."
                return ozonetel_speak_and_listen(response_text)

            intent = classify_intent_with_llm(user_speech_result)
            session_data['intent'] = intent
            logger.info(f"Classified intent for '{user_speech_result}': {intent}")

            if intent == 'query_emi':
                session_data['stage'] = 'ask_account_id'
                VOICE_SESSION_STORE[call_sid] = session_data
                response_text = "To fetch your EMI details, please speak or enter your 10-digit account ID."
                return ozonetel_speak_and_listen(response_text)
            elif intent == 'live_agent_request':
                session_data['stage'] = 'handoff_init'
                VOICE_SESSION_STORE[call_sid] = session_data
                logger.info(f"Handoff requested for CallSid {call_sid}. Creating TaskRouter Task.")
                
                # --- TaskRouter Handoff Logic ---
                global WORKSPACE_INFO
                if not WORKSPACE_INFO:
                    # Attempt to get workspace info if not already loaded
                    client = build_twilio_client()
                    WORKSPACE_INFO = taskrouter_config_setup(client)

                if not WORKSPACE_INFO or not WORKSPACE_INFO.workflow_sid or not WORKSPACE_INFO.workspace_sid:
                    logger.error("TaskRouter Workspace info not available for handoff.")
                    response_text = "I am unable to connect you to an agent right now. Please try again later."
                    return ozonetel_speak_and_hangup(response_text)

                try:
                    # Create TaskRouter Task
                    customer_phone_number = from_number # The number calling in
                    task_attributes = {
                        "customer_call_sid": call_sid,
                        "customer_phone": customer_phone_number,
                        "selected_product": "VoiceHandoff", # Specific product for voice handoff
                        "type": "voice_handoff_request",
                        "direction": "inbound"
                    }
                    twilio_client = build_twilio_client()
                    task = twilio_client.taskrouter.workspaces(WORKSPACE_INFO.workspace_sid).tasks.create(
                        workflow_sid=WORKSPACE_INFO.workflow_sid,
                        attributes=json.dumps(task_attributes),
                        task_channel="voice" # Assuming a 'voice' TaskChannel exists in TaskRouter
                    )
                    session_data['task_sid'] = task.sid
                    VOICE_SESSION_STORE[call_sid] = session_data
                    logger.info(f"TaskRouter Task {task.sid} created for CallSid {call_sid}")
                    
                    # Store unresolved chat for future analysis/follow-up
                    with Session() as db_session:
                        unresolved_chat = UnresolvedChat(
                            customer_id=session_data.get('customer_id', 'N/A'), # Use actual customer_id if known, else 'N/A'
                            account_id=session_data.get('account_id', 'N/A'),
                            session_id=str(current_session_id),
                            summary=f"Voice call handoff requested by {from_number} for general assistance. Initial query: '{user_speech_result}'",
                            embedding_vector=[] # Placeholder for actual embedding
                        )
                        db_session.add(unresolved_chat)
                        db_session.commit()
                        logger.info(f"Saved unresolved chat summary for CallSid {call_sid}")

                    response_text = "Please wait while I connect you to the next available agent."
                    return ozonetel_speak_and_listen(response_text) # Keep listening in case of agent busy
                except Exception as e:
                    logger.error(f"Error creating TaskRouter Task: {e}", exc_info=True)
                    response_text = "I'm sorry, I encountered an issue trying to connect you to an agent. Please try again later."
                    return ozonetel_speak_and_hangup(response_text)

            else: # Unclear intent
                response_text = "I'm sorry, I didn't understand your request. You can ask about your EMI or say 'connect to agent'."
                return ozonetel_speak_and_listen(response_text)

        # Awaiting Account ID
        elif session_data['stage'] == 'ask_account_id':
            account_id = extract_digits_from_speech(user_speech_result)
            logger.info(f"Extracted Account ID: {account_id}")

            with Session() as db_session:
                # Assuming database.py has fetch_customer_by_account that joins Customer and CustomerAccount
                customer_data = db_session.query(Customer, EMI).join(EMI, Customer.customer_id == EMI.customer_id).filter(Customer.account_id == account_id).first()
                
            if customer_data:
                customer, emi_record = customer_data
                session_data['customer_id'] = customer.customer_id
                session_data['account_id'] = customer.account_id
                session_data['phone_number'] = customer.phone_number
                session_data['stage'] = 'otp_pending'
                VOICE_SESSION_STORE[call_sid] = session_data

                # Send OTP via Twilio
                if send_sms_otp(customer.phone_number, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_MESSAGING_SERVICE_SID):
                    logger.info(f"OTP sent to {hide_number(customer.phone_number)} for account_id={customer.account_id}")
                    response_text = f"I have sent a 6-digit OTP to your registered mobile number ending in {customer.phone_number[-4:]}. Please speak the OTP now."
                    return ozonetel_speak_and_listen(response_text)
                else:
                    response_text = "There was an issue sending the OTP. Please try again later."
                    return ozonetel_speak_and_hangup(response_text)
            else:
                response_text = "I could not find an account with that ID. Please try again or say 'connect to agent'."
                return ozonetel_speak_and_listen(response_text)

        # Awaiting OTP
        elif session_data['stage'] == 'otp_pending':
            otp_code = extract_digits_from_speech(user_speech_result)
            logger.info(f"Extracted OTP: {otp_code}")

            if verify_otp_code(session_data['phone_number'], otp_code):
                session_data['stage'] = 'verified'
                VOICE_SESSION_STORE[call_sid] = session_data
                logger.info(f"🏆 OTP verified for phone {session_data['phone_number']}")

                # Fetch EMI details
                with Session() as db_session:
                    emi_record = db_session.query(EMI).filter_by(
                        loan_id=session_data['account_id'] # Assuming loan_id is same as account_id for EMI, adjust if needed
                    ).order_by(EMI.created_at.desc()).first()
                    customer_info = db_session.query(Customer).filter_by(customer_id=session_data['customer_id']).first()

                if emi_record and customer_info:
                    response_text = generate_emi_response_with_rag(emi_record, customer_info)
                    return ozonetel_speak_and_hangup(response_text + " Thank you for calling.")
                else:
                    response_text = "I could not retrieve your EMI details. Please contact support if the issue persists. Thank you."
                    return ozonetel_speak_and_hangup(response_text)
            else:
                response_text = "That OTP is incorrect. Please try again or say 'connect to agent'."
                return ozonetel_speak_and_listen(response_text)

        # Default fallback for unhandled stages
        response_text = "I'm sorry, something went wrong. Please call again."
        return ozonetel_speak_and_hangup(response_text)

    except Exception as e:
        logger.error(f"Error in handle_ozonetel_voice_call for CallSid {call_sid}: {e}", exc_info=True)
        response_text = "I apologize, an unexpected error occurred. Please try again later."
        return ozonetel_speak_and_hangup(response_text)
    finally:
        # Log bot's response
        if 'response_text' in locals() and response_text:
            with Session() as db_session:
                db_session.add(ClientInteraction(
                    session_id=current_session_id,
                    customer_id=session_data.get('customer_id', 'N/A'),
                    sender='bot',
                    message_text=response_text,
                    stage=session_data['stage'],
                    intent=session_data.get('intent')
                ))
                db_session.commit()

@app.route('/assignment', methods=['POST'])
def handle_taskrouter_assignment():
    """
    Handles the assignment callback from Twilio TaskRouter.
    When a Task is assigned to a Worker, TaskRouter sends a POST request here.
    """
    try:
        assignment_info = request.get_json() # TaskRouter sends JSON payload
        if not assignment_info: # Fallback if content-type is not application/json
            assignment_info = request.form.to_dict()

        logger.info(f"Received TaskRouter Assignment: {assignment_info}")

        task_attributes = json.loads(assignment_info.get('TaskAttributes', '{}'))
        worker_attributes = json.loads(assignment_info.get('WorkerAttributes', '{}'))
        worker_contact_uri = worker_attributes.get('contact_uri') # This is the agent's phone number from TaskRouter worker config
        customer_call_sid = task_attributes.get('customer_call_sid') # Original CallSid from Ozonetel

        if not worker_contact_uri or not customer_call_sid:
            logger.error("Missing worker_contact_uri or customer_call_sid in assignment.")
            return jsonify({"instruction": "reject"}) # Reject assignment if critical info is missing

        # --- Instruct Ozonetel to bridge the call ---
        # This is the point where we tell Ozonetel to connect the original call_sid
        # to the agent_phone_number (worker_contact_uri).
        # The specific JSON format for Ozonetel's 'dial' instruction needs to be confirmed
        # with Ozonetel's API documentation.
        
        # Example Ozonetel 'dial' instruction (ASSUMPTION!)
        ozonetel_dial_agent_response = ozonetel_dial_agent(worker_contact_uri)
        
        # TaskRouter expects a JSON response with instructions for the reservation.
        # If we successfully instruct Ozonetel to bridge, we accept the reservation.
        # The instruction for TaskRouter here is typically 'accept' or 'call'.
        # For an external voice platform like Ozonetel, 'accept' is more appropriate
        # as Twilio TaskRouter is not directly handling the call itself after assignment.
        return jsonify({"instruction": "accept"})
        
    except Exception as e:
        logger.error(f"Error in TaskRouter assignment handler: {e}", exc_info=True)
        return jsonify({"instruction": "reject"}) # Reject assignment on error

@app.route('/events', methods=['POST'])
def handle_taskrouter_events():
    """
    Receives events from Twilio TaskRouter.
    Used for logging and debugging TaskRouter's activity (e.g., Task created, Worker status change).
    """
    logger.info(f"Received TaskRouter Event: {request.form.to_dict()}")
    return "", 200 # Acknowledge receipt


@app.route('/worker_activity_update', methods=['POST'])
def update_worker_activity():
    """
    Endpoint for agents to update their activity via SMS or a simple UI.
    Example: Agent sends SMS "available" to this number.
    This endpoint assumes Twilio will forward an SMS to this webhook.
    """
    from_number = request.form.get('From') # Agent's phone number
    message_body = request.form.get('Body', '').strip().lower()

    if not from_number or not message_body:
        logger.warning("Missing 'From' or 'Body' in worker_activity_update request.")
        return "Missing parameters", 400

    global WORKSPACE_INFO
    if not WORKSPACE_INFO:
        client = build_twilio_client()
        WORKSPACE_INFO = taskrouter_config_setup(client) # Ensure workspace info is loaded

    if not WORKSPACE_INFO or not WORKSPACE_INFO.workers or not WORKSPACE_INFO.activities:
        logger.error("TaskRouter Workspace info or worker/activity data not available for activity update.")
        return "TaskRouter not configured correctly.", 500

    # Ensure phone numbers are consistent (e.g., +91 format)
    # You might need to normalize from_number if it comes in different formats
    worker_sid = WORKSPACE_INFO.workers.get(from_number) # Get worker SID by their phone number
    if not worker_sid:
        logger.warning(f"No worker found for phone number: {from_number}. Registered workers: {WORKSPACE_INFO.workers.keys()}")
        # You might want to send an SMS reply here, e.g., "You are not a registered worker."
        return f"Worker with number {from_number} not found.", 404

    target_activity_sid = None
    if message_body == 'available':
        target_activity_sid = WORKSPACE_INFO.activities.get('Available').sid
    elif message_body == 'offline':
        target_activity_sid = WORKSPACE_INFO.activities.get('Offline').sid
    elif message_body == 'busy':
        target_activity_sid = WORKSPACE_INFO.activities.get('Busy').sid
    else:
        logger.info(f"Unknown activity command from {from_number}: '{message_body}'")
        # Send SMS back: "Invalid command. Use 'available', 'offline', or 'busy'."
        return "Invalid command. Please use 'available', 'offline', or 'busy'.", 400

    if target_activity_sid:
        try:
            twilio_client = build_twilio_client()
            twilio_client.taskrouter.workspaces(WORKSPACE_INFO.workspace_sid)\
                          .workers(worker_sid).update(activity_sid=target_activity_sid)
            logger.info(f"Worker {worker_sid} ({from_number}) activity set to {message_body.capitalize()}")
            # Send SMS back to agent: "Your status is now {message_body.capitalize()}."
            return f"Your status is now {message_body.capitalize()}.", 200
        except Exception as e:
            logger.error(f"Error updating worker activity for {from_number}: {e}", exc_info=True)
            return "Failed to update your status. Please try again.", 500
    return "No activity specified or invalid.", 400


if __name__ == '__main__':
    # Initialize database and sample data on startup
    with app.app_context():
        setup_database()

    # Load TaskRouter config on startup
    TASKROUTER_CLIENT = build_twilio_client() # Initialize once
    WORKSPACE_INFO = taskrouter_config_setup(TASKROUTER_CLIENT)

    app.run(debug=os.getenv("FLASK_DEBUG") == "True", host='0.0.0.0', port=5000)
