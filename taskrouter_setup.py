import os
import json
import logging
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST")
ALICE_NUMBER = os.getenv("ALICE_NUMBER")
BOB_NUMBER = os.getenv("BOB_NUMBER")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

WORKSPACE_NAME = 'Financial Voice Bot Workspace'

def first(items):
    """Helper to get the first item from a list or None."""
    return items[0] if items else None

def build_client():
    """Builds and returns a Twilio REST Client."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not set in .env")
        raise ValueError("Twilio credentials missing.")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def get_activities_dict(client, workspace_sid):
    """Fetches and returns a dictionary of activities by friendly name."""
    logger.info(f"Fetching activities for workspace SID: {workspace_sid}")
    activities = client.taskrouter.workspaces(workspace_sid).activities.list()
    if not activities:
        logger.warning("No activities found in the workspace. This might indicate an issue.")
    return {activity.friendly_name: activity for activity in activities}

class WorkspaceInfo:
    """A class to hold key TaskRouter SIDs and objects."""
    def __init__(self, workspace, workflow, activities, workers):
        self.workflow_sid = workflow.sid
        self.workspace_sid = workspace.sid
        self.activities = activities # Dict of activity friendly_name -> activity object
        self.post_work_activity_sid = activities.get('Available').sid if 'Available' in activities else None
        self.workers = workers # Dict of worker_number -> worker_sid

    def __repr__(self):
        return (f"<WorkspaceInfo(workspace_sid='{self.workspace_sid}', "
                f"workflow_sid='{self.workflow_sid}', "
                f"activities_count={len(self.activities)}, "
                f"workers_count={len(self.workers)})>")

# Cache to avoid re-creating TaskRouter resources during development if possible
_WORKSPACE_CACHE = {}

def setup(client):
    """
    Sets up or retrieves the TaskRouter Workspace, Workers, Queues, and Workflow.
    This function is idempotent for multiple calls within the same process.
    """
    global _WORKSPACE_CACHE

    if 'WORKSPACE_INFO' in _WORKSPACE_CACHE:
        logger.info("TaskRouter Workspace info already in cache. Reusing.")
        return _WORKSPACE_CACHE['WORKSPACE_INFO']

    logger.info("Setting up TaskRouter Workspace...")
    workspace = create_workspace(client)
    logger.info(f"Workspace '{workspace.friendly_name}' created/verified with SID: {workspace.sid}")

    activities = get_activities_dict(client, workspace.sid)
    # Ensure 'Available', 'Offline', 'Busy', 'Unavailable' activities exist
    required_activities = ['Available', 'Offline', 'Busy', 'Unavailable']
    for req_act in required_activities:
        if req_act not in activities:
            logger.error(f"Required activity '{req_act}' not found in workspace. Attempting to create.")
            try:
                # 'Available' activity needs to be marked as available=True
                is_available = True if req_act == 'Available' else False
                new_activity = client.taskrouter.workspaces(workspace.sid).activities.create(friendly_name=req_act, available=is_available)
                activities[req_act] = new_activity
                logger.info(f"Created missing activity: {req_act}")
            except Exception as e:
                logger.error(f"Failed to create activity {req_act}: {e}")
                # This is a critical failure, as essential activities are missing.
                raise RuntimeError(f"Failed to set up TaskRouter: Missing required activity {req_act}")


    workers = create_workers(client, workspace, activities)
    logger.info(f"Workers created/verified: {workers}")

    queues = create_task_queues(client, workspace, activities)
    logger.info(f"Task Queues created/verified: {', '.join(queues.keys())}")

    workflow = create_workflow(client, workspace, queues)
    logger.info(f"Workflow '{workflow.friendly_name}' created/verified with SID: {workflow.sid}")

    workspace_info = WorkspaceInfo(workspace, workflow, activities, workers)
    _WORKSPACE_CACHE['WORKSPACE_INFO'] = workspace_info
    logger.info("TaskRouter setup complete.")
    return workspace_info

def create_workspace(client):
    """Creates or retrieves a TaskRouter Workspace, deleting existing ones with same name."""
    existing_workspaces = client.taskrouter.workspaces.list(friendly_name=WORKSPACE_NAME)
    for ws in existing_workspaces:
        logger.info(f"Deleting existing workspace '{ws.friendly_name}' with SID: {ws.sid}")
        try:
            client.taskrouter.workspaces(ws.sid).delete()
        except Exception as e:
            logger.warning(f"Failed to delete workspace {ws.sid}: {e}. It might be in use or already deleted.")


    events_callback = HOST + '/events' # Endpoint for TaskRouter events
    logger.info(f"Creating new workspace '{WORKSPACE_NAME}' with event callback: {events_callback}")
    
    # template=None to ensure default activities are created
    return client.taskrouter.workspaces.create(
        friendly_name=WORKSPACE_NAME,
        event_callback_url=events_callback,
        template=None
    )

def create_workers(client, workspace, activities):
    """Creates or retrieves sample Workers (Alice and Bob and LiveAgent_1)."""
    if not ALICE_NUMBER or not BOB_NUMBER:
        logger.error("ALICE_NUMBER or BOB_NUMBER not set in .env. Cannot create workers.")
        raise ValueError("Agent numbers missing.")

    workers_sids = {}
    
    # Define workers and their attributes
    worker_definitions = [
        {'friendly_name': 'Alice', 'number': ALICE_NUMBER, 'products': ["ProgrammableVoice"], 'initial_activity': 'Available'},
        {'friendly_name': 'Bob', 'number': BOB_NUMBER, 'products': ["ProgrammableSMS", "GeneralSupport"], 'initial_activity': 'Available'},
        # LiveAgent_1 also maps to Alice's number, but has 'LiveAgent' and 'VoiceHandoff' skills
        {'friendly_name': 'LiveAgent_1', 'number': ALICE_NUMBER, 'products': ["LiveAgent", "VoiceHandoff"], 'initial_activity': 'Available'}
    ]

    for worker_def in worker_definitions:
        friendly_name = worker_def['friendly_name']
        number = worker_def['number']
        products = worker_def['products']
        initial_activity = worker_def['initial_activity']
        
        # Check if worker already exists and delete if so, for idempotency
        existing_workers = client.taskrouter.workspaces(workspace.sid).workers.list(friendly_name=friendly_name)
        for w in existing_workers:
            logger.info(f"Deleting existing worker '{w.friendly_name}' with SID: {w.sid}")
            try:
                client.taskrouter.workspaces(workspace.sid).workers(w.sid).delete()
            except Exception as e:
                logger.warning(f"Failed to delete worker {w.sid}: {e}. It might be in use or already deleted.")


        attributes = {
            "products": products,
            "contact_uri": number # The phone number TaskRouter will use to reach the worker
        }
        
        # Create the worker, set initial activity
        worker = client.taskrouter.workspaces(workspace.sid).workers.create(
            friendly_name=friendly_name,
            attributes=json.dumps(attributes),
            activity_sid=activities[initial_activity].sid
        )
        logger.info(f"Created worker '{friendly_name}' (SID: {worker.sid}) with attributes: {attributes}")
        workers_sids[number] = worker.sid # Store by phone number for easy lookup

    return workers_sids

def create_task_queues(client, workspace, activities):
    """Creates or retrieves Task Queues."""
    queues_dict = {}
    
    queue_definitions = [
        {'friendly_name': 'Default', 'target_workers': '1==1'},
        {'friendly_name': 'SMS', 'target_workers': '"ProgrammableSMS" in products'},
        {'friendly_name': 'Voice', 'target_workers': '"ProgrammableVoice" in products'},
        {'friendly_name': 'LiveAgent_Handoff', 'target_workers': '"LiveAgent" in products AND "VoiceHandoff" in products'}, # For voice handoff
    ]

    for q_def in queue_definitions:
        friendly_name = q_def['friendly_name']
        target_workers_expression = q_def['target_workers']

        # Check if queue already exists and delete if so
        existing_queues = client.taskrouter.workspaces(workspace.sid).task_queues.list(friendly_name=friendly_name)
        for q in existing_queues:
            logger.info(f"Deleting existing Task Queue '{q.friendly_name}' with SID: {q.sid}")
            try:
                client.taskrouter.workspaces(workspace.sid).task_queues(q.sid).delete()
            except Exception as e:
                logger.warning(f"Failed to delete task queue {q.sid}: {e}. It might be in use or already deleted.")


        # assignment_activity_sid: The activity a worker will enter when assigned a task from this queue.
        # 'Unavailable' is common for agents actively handling a task.
        queue = client.taskrouter.workspaces(workspace.sid).task_queues.create(
            friendly_name=friendly_name,
            assignment_activity_sid=activities['Unavailable'].sid,
            target_workers=target_workers_expression
        )
        logger.info(f"Created Task Queue '{friendly_name}' (SID: {queue.sid}) with target_workers: '{target_workers_expression}'")
        queues_dict[friendly_name.lower().replace(" ", "_")] = queue # Use snake_case for dict keys

    return queues_dict

def create_workflow(client, workspace, queues):
    """Creates or retrieves the Workflow."""
    
    # Delete existing workflows with the same name for idempotency
    existing_workflows = client.taskrouter.workspaces(workspace.sid).workflows.list(friendly_name='Sales')
    for wf in existing_workflows:
        logger.info(f"Deleting existing Workflow '{wf.friendly_name}' with SID: {wf.sid}")
        try:
            client.taskrouter.workspaces(workspace.sid).workflows(wf.sid).delete()
        except Exception as e:
            logger.warning(f"Failed to delete workflow {wf.sid}: {e}. It might be in use or already deleted.")

    # Define targets for filters
    default_target = {
        'queue': queues['default'].sid,
        'priority': 5,
        'timeout': 30
    }

    # Target for the LiveAgent_Handoff queue
    voice_handoff_target = {
        'queue': queues['liveagent_handoff'].sid,
        'priority': 1, # Higher priority for live agent requests
        'timeout': 60 # Longer timeout for agent to answer the bridged call
    }
    
    # Define filters
    # The 'VoiceHandoff' filter matches tasks specifically for voice bot handoff
    voice_handoff_filter = {
        'filter_friendly_name': 'Voice Handoff Filter',
        'expression': 'type=="voice_handoff_request"', # Task attribute 'type' will be 'voice_handoff_request'
        'targets': [voice_handoff_target, default_target] # Try voice handoff queue, then default
    }

    config = {
        'task_routing': {
            'filters': [voice_handoff_filter], # Add other filters if you have more task types
            'default_filter': default_target # Tasks that don't match any filter go here
        }
    }

    assignment_callback_url = HOST + '/assignment' # Your Flask endpoint for Task assignments
    fallback_assignment_callback_url = HOST + '/assignment' # Fallback if primary fails

    logger.info(f"Creating Workflow 'Sales' with assignment callback: {assignment_callback_url}")
    return client.taskrouter.workspaces(workspace.sid).workflows.create(
        friendly_name='Sales', # Workflow name
        assignment_callback_url=assignment_callback_url,
        fallback_assignment_callback_url=fallback_assignment_callback_url,
        task_reservation_timeout=20, # How long a task waits for a worker to accept
        configuration=json.dumps(config)
    )

if __name__ == '__main__':
    # This block runs when taskrouter_setup.py is executed directly
    logger.info("Starting TaskRouter setup script...")
    try:
        twilio_client = build_client()
        setup(twilio_client)
        logger.info("TaskRouter setup script finished successfully.")
    except Exception as e:
        logger.error(f"TaskRouter setup script failed: {e}", exc_info=True)