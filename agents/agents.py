# type: ignore
from common import common
from core.core import LlmChatAgent
from tools.get_vms import *
from tools.execute_on_vm import *
from tools.triage import triage_instructions
from tools.analyse_webpage import *
from tools.summarize import *
from tools.vm_creator import *
from tools.analyse_security import *
from tools.vm_operation import *
from tools.tech_support import *
from tools.analyse_vm import *

context_variables = {
    "webpage_summary": "",
    "requirements_summary": "",
    "vm_id": "",
    "vm_details": "",
}


def transfer_to_get_vms(context_variables):
    """Call this function if the user is asking you to list all VMs, or list a specific VM or list the details of a specific VM or all VMs."""
    return get_vms_agent


def transfer_to_execute_on_vms(context_variables):
    """Call this function if the user is asking you to execute a command on a VM."""
    return execute_on_vms_agent


def transfer_back_to_triage(context_variables):
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent


def transfer_to_analyse_webpage_agent(context_variables):
    """Call this function if the user is asking you to analyse a webpage or code in a webpage."""
    return analyse_webpage_agent


def transfer_to_create_vm_agent(context_variables):
    """Call this function if the user is asking you to create a VM."""
    return create_vm_agent


def transfer_to_summarize_agent(context_variables):
    """Call this function if there is nothing else to do."""
    return summarize_agent


def transfer_to_analyse_security_agent(context_variables):
    """Call this function if the user is asking you to analyse the security, vulnerabilities, or updates of a VM.
    You will need the VM ID or VM Name to do this. check the context or history of the conversation for this information.
    """
    return analyse_security_agent


def transfer_to_operation_vm_agent(context_variables):
    """Call this function if the user is asking you to start, stop, suspend, resume, pause, or delete a VM.
    You will need the VM ID or VM Name to do this. check the context or history of the conversation for this information.
    """
    return operation_vm_agent


def transfer_to_tech_support_agent(context_variables):
    """Call this function if the user is asking you to provide technical support or any generic questions that are not handled by the other agents.
    for example, how do I run this, or how do I fix this, or check for errors on the vm, or how do I update this, etc.
    """
    return tech_support_agent


def transfer_to_analyse_vm_agent(context_variables):
    """Call this function if the user is asking you to analyse the VM, get a health check, or get the ocr of a screenshot or just a screenshot.
    You will need the VM ID or VM Name to do this, if the state is stopped, you will need to start the vm first. check the context or history of the conversation for this information.
    """
    return analyse_vm_agent


triage_agent = LlmChatAgent(
    name="Triage Agent",
    model=common.DEFAULT_MODEL,
    instructions=triage_instructions,
    parallel_tool_calls=False,
    functions=[
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
        transfer_to_tech_support_agent,
        transfer_to_analyse_vm_agent,
    ],
)

get_vms_agent = LlmChatAgent(
    name="Get VMs Agent",
    model=common.DEFAULT_MODEL,
    instructions=GET_VMS_AGENT_PROMPT,
    functions=[
        agent_get_vms_tool,
        agent_get_vm_tool,
        transfer_back_to_triage,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
        transfer_to_tech_support_agent,
    ],
)

execute_on_vms_agent = LlmChatAgent(
    name="Execute on VMs Agent",
    model=common.DEFAULT_MODEL,
    instructions="Helps the user to execute commands on a VM. it will always respond with a json object with the following keys: status, message, context_variables, and once you are done, return to the triage agent.",
    functions=[
        agent_execute_on_vm_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
        transfer_to_tech_support_agent,
    ],
)

analyse_webpage_agent = LlmChatAgent(
    name="Analyse Webpage Agent",
    model=common.DEFAULT_MODEL,
    instructions=ANALYSE_WEB_PAGE_PROMPT,
    functions=[
        agent_analyse_webpage_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
    ],
)

create_vm_agent = LlmChatAgent(
    name="Create VM Agent",
    model=common.DEFAULT_MODEL,
    instructions=VM_CREATOR_PROMPT,
    functions=[
        create_vm_tool,
        agent_clone_vm_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
    ],
)

summarize_agent = LlmChatAgent(
    name="Summarize Agent",
    model=common.DEFAULT_MODEL,
    instructions=SUMMARIZE_PROMPT,
    functions=[
        agent_summarize_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
    ],
)

analyse_security_agent = LlmChatAgent(
    name="Analyse Security Agent",
    model=common.DEFAULT_MODEL,
    instructions=ANALYSE_SECURITY_PROMPT,
    functions=[
        agent_analyse_security_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_operation_vm_agent,
    ],
)

operation_vm_agent = LlmChatAgent(
    name="Operation VM Agent",
    model=common.DEFAULT_MODEL,
    instructions=VM_OPERATION_PROMPT,
    functions=[
        agent_start_vm_tool,
        agent_stop_vm_tool,
        agent_suspend_vm_tool,
        agent_resume_vm_tool,
        agent_pause_vm_tool,
        agent_delete_vm_tool,
        agent_restart_vm_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
    ],
)

tech_support_agent = LlmChatAgent(
    name="Tech Support Agent",
    model=common.DEFAULT_MODEL,
    instructions=TECH_SUPPORT_PROMPT,
    functions=[
        agent_tech_support_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
        transfer_to_analyse_security_agent,
        transfer_to_operation_vm_agent,
    ],
)

analyse_vm_agent = LlmChatAgent(
    name="Analyse VM Agent",
    model=common.DEFAULT_MODEL,
    instructions=ANALYSE_VM_PROMPT,
    functions=[
        agent_get_ocr_from_screenshot_tool,
        agent_get_health_check_tool,
        transfer_back_to_triage,
        transfer_to_get_vms,
        transfer_to_execute_on_vms,
        transfer_to_analyse_webpage_agent,
        transfer_to_create_vm_agent,
        transfer_to_summarize_agent,
    ],
)
