from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgent,
    LlmChatResult,
    LlmChatAgentResponse,
    AgentFunctionDescriptor,
    AttachmentContextVariable,
    AttachmentType,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.log_service import LogService
from pd_ai_agent_core.services.ocr_service import OCRService

from pd_ai_agent_core.services.vm_datasource_service import VmDatasourceService
from pd_ai_agent_core.messages import (
    create_agent_function_call_chat_message,
    create_clean_agent_function_call_chat_message,
)
import json
import logging
from pd_ai_agent_core.parallels_desktop.get_vms import get_vm
from pd_ai_agent_core.parallels_desktop.execute_on_vm import execute_on_vm
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.parallels_desktop.get_vm_screenshot import get_vm_screenshot
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
    OCR_SERVICE_NAME,
)
import openai
from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgentResponseAction,
)

logger = logging.getLogger(__name__)


def ANALYSE_SUPPORT_PROMPT(os_version: str, ocr_text: str) -> str:
    return (
        """You are a seasoned AI agent that can analyse extracted text from a screenshot.
Please pay close attention to the operating system of the vm so you can better understand the context of the screenshot.
Try to give as much information as possible to the user about the screenshot as it will help the user to understand the screenshot better and might be used by other agents in the chain.
Always include the screenshot data in your response.

The ocr text is:"""
        + ocr_text
        + """

The os version is:"""
        + os_version
    )


def SCREENSHOT_OCR_PROMPT(context_variables) -> str:
    result = """You are a seasoned AI agent that can extract text from virtual machine screenshots.
You will be responsible for taking the screenshot for the user and then extracting the text from it.

You will need to know the virtual machine id so you can take the screenshot. if this is not provided, you should ask the user for it.
Pay close attention to the operating system of the vm so you can better understand the context of the screenshot.

Try to give as much information as possible to the user about the screenshot as it will help the user to understand the screenshot better and might be used by other agents in the chain.

if you spot any errors in the screenshot, please let the user know about it in a very friendly but constructive way as this information will be used by other agents in the chain.
"""
    result = result.format(context_variables=context_variables)
    return result


SCREENSHOT_OCR_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to take a screenshot of a vm or analyzer the vm screen or screenshot.
"""


class ScreenshotOcrAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Screenshot OCR Agent",
            instructions=SCREENSHOT_OCR_PROMPT,
            description="This agent is responsible for taking a screenshot of a vm or analyzer the vm screen or screenshot.",
            functions=[self.screenshot_ocr],  # type: ignore
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+Cgo8c3ZnCiAgIHdpZHRoPSI4MDBweCIKICAgaGVpZ2h0PSI4MDBweCIKICAgdmlld0JveD0iMCAwIDUwLjggNTAuOCIKICAgdmVyc2lvbj0iMS4xIgogICBpZD0ic3ZnMSIKICAgc29kaXBvZGk6ZG9jbmFtZT0ic2NyZWVuc2hvdC10aWxlLW5vcm9vdC1zdmdyZXBvLWNvbS5zdmciCiAgIGlua3NjYXBlOnZlcnNpb249IjEuNCAoZTdjM2ZlYjEsIDIwMjQtMTAtMDkpIgogICB4bWxuczppbmtzY2FwZT0iaHR0cDovL3d3dy5pbmtzY2FwZS5vcmcvbmFtZXNwYWNlcy9pbmtzY2FwZSIKICAgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIgogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxkZWZzCiAgICAgaWQ9ImRlZnMxIiAvPgogIDxzb2RpcG9kaTpuYW1lZHZpZXcKICAgICBpZD0ibmFtZWR2aWV3MSIKICAgICBwYWdlY29sb3I9IiNmZmZmZmYiCiAgICAgYm9yZGVyY29sb3I9IiMwMDAwMDAiCiAgICAgYm9yZGVyb3BhY2l0eT0iMC4yNSIKICAgICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTpwYWdlb3BhY2l0eT0iMC4wIgogICAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjAiCiAgICAgaW5rc2NhcGU6ZGVza2NvbG9yPSIjZDFkMWQxIgogICAgIGlua3NjYXBlOnpvb209IjAuODkxODM4NDMiCiAgICAgaW5rc2NhcGU6Y3g9IjM3MC41ODI4MiIKICAgICBpbmtzY2FwZTpjeT0iNDY1Ljg5MTU2IgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTIwMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctaGVpZ2h0PSIxMTg2IgogICAgIGlua3NjYXBlOndpbmRvdy14PSIwIgogICAgIGlua3NjYXBlOndpbmRvdy15PSIyNSIKICAgICBpbmtzY2FwZTp3aW5kb3ctbWF4aW1pemVkPSIwIgogICAgIGlua3NjYXBlOmN1cnJlbnQtbGF5ZXI9ImcxIiAvPgogIDxnCiAgICAgZmlsbD0ibm9uZSIKICAgICBzdHJva2U9IiMwMTAwMDAiCiAgICAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogICAgIHN0cm9rZS1saW5lam9pbj0icm91bmQiCiAgICAgc3Ryb2tlLXdpZHRoPSIzLjE3NSIKICAgICBpZD0iZzEiPgogICAgPHBhdGgKICAgICAgIHN0eWxlPSJmaWxsOiMwMTAwMDA7c3Ryb2tlOm5vbmUiCiAgICAgICBkPSJNIDcuOTM3NSw2LjM0OTYwOTQgQSAxLjU4NzY1ODcsMS41ODc2NTg3IDAgMCAwIDYuMzQ5NjA5NCw3LjkzNzUgViAxOS44NDM3NSBBIDEuNTg3NSwxLjU4NzUgMCAwIDAgNy45Mzc1LDIxLjQyOTY4NyAxLjU4NzUsMS41ODc1IDAgMCAwIDkuNTI1MzkwNiwxOS44NDM3NSBWIDkuNTI1MzkwNiBIIDE5Ljg0Mzc1IEEgMS41ODc1LDEuNTg3NSAwIDAgMCAyMS40MzE2NDEsNy45Mzc1IDEuNTg3NSwxLjU4NzUgMCAwIDAgMTkuODQzNzUsNi4zNDk2MDk0IFogbSAyMy4wMTc1NzgsMCBBIDEuNTg3NSwxLjU4NzUgMCAwIDAgMjkuMzY5MTQxLDcuOTM3NSAxLjU4NzUsMS41ODc1IDAgMCAwIDMwLjk1NTA3OCw5LjUyNTM5MDYgSCA0MS4yNzUzOTEgViAxOS44NDM3NSBhIDEuNTg3NSwxLjU4NzUgMCAwIDAgMS41ODU5MzcsMS41ODc4OTEgMS41ODc1LDEuNTg3NSAwIDAgMCAxLjU4Nzg5MSwtMS41ODc4OTEgViA3LjkzNzUgQSAxLjU4NzY1ODcsMS41ODc2NTg3IDAgMCAwIDQyLjg2MTMyOCw2LjM0OTYwOTQgWiBNIDcuOTM3NSwyOS4zNjkxNDEgYSAxLjU4NzUsMS41ODc1IDAgMCAwIC0xLjU4Nzg5MDYsMS41ODU5MzcgdiAxMS45MDYyNSBBIDEuNTg3NjU4NywxLjU4NzY1ODcgMCAwIDAgNy45Mzc1LDQ0LjQ0OTIxOSBIIDE5Ljg0Mzc1IEEgMS41ODc1LDEuNTg3NSAwIDAgMCAyMS40Mjk2ODcsNDIuODYxMzI4IDEuNTg3NSwxLjU4NzUgMCAwIDAgMTkuODQzNzUsNDEuMjc1MzkxIEggOS41MjUzOTA2IFYgMzAuOTU1MDc4IEEgMS41ODc1LDEuNTg3NSAwIDAgMCA3LjkzNzUsMjkuMzY5MTQxIFogbSAzNC45MjM4MjgsMCBhIDEuNTg3NSwxLjU4NzUgMCAwIDAgLTEuNTg1OTM3LDEuNTg1OTM3IFYgNDEuMjc1MzkxIEggMzAuOTU1MDc4IGEgMS41ODc1LDEuNTg3NSAwIDAgMCAtMS41ODU5MzcsMS41ODU5MzcgMS41ODc1LDEuNTg3NSAwIDAgMCAxLjU4NTkzNywxLjU4Nzg5MSBoIDExLjkwNjI1IGEgMS41ODc2NTg3LDEuNTg3NjU4NyAwIDAgMCAxLjU4Nzg5MSwtMS41ODc4OTEgdiAtMTEuOTA2MjUgYSAxLjU4NzUsMS41ODc1IDAgMCAwIC0xLjU4Nzg5MSwtMS41ODU5MzcgeiIKICAgICAgIGlkPSJwYXRoMSIgLz4KICAgIDxwYXRoCiAgICAgICBzdHlsZT0iZmlsbDojMDEwMDAwO3N0cm9rZTpub25lIgogICAgICAgZD0ibSAyNS40MDAzOTEsMTUuMDgyMDMxIGMgLTUuNjc5OTQ3LDAgLTEwLjMxODM2LDQuNjM4NDEzIC0xMC4zMTgzNiwxMC4zMTgzNiAwLDUuNjc5OTQ2IDQuNjM4NDEzLDEwLjMxODM1OSAxMC4zMTgzNiwxMC4zMTgzNTkgNS42Nzk5NDYsMCAxMC4zMTgzNTksLTQuNjM4NDEzIDEwLjMxODM1OSwtMTAuMzE4MzU5IDAsLTUuNjc5OTQ3IC00LjYzODQxMywtMTAuMzE4MzYgLTEwLjMxODM1OSwtMTAuMzE4MzYgeiBtIDAsMy4xNzM4MjggYyAzLjk2NDA0OSwwIDcuMTQyNTc4LDMuMTgwNDgyIDcuMTQyNTc4LDcuMTQ0NTMyIDAsMy45NjQwNDkgLTMuMTc4NTI5LDcuMTQyNTc4IC03LjE0MjU3OCw3LjE0MjU3OCAtMy45NjQwNSwwIC03LjE0NDUzMiwtMy4xNzg1MjkgLTcuMTQ0NTMyLC03LjE0MjU3OCAwLC0zLjk2NDA1IDMuMTgwNDgyLC03LjE0NDUzMiA3LjE0NDUzMiwtNy4xNDQ1MzIgeiIKICAgICAgIGlkPSJjaXJjbGUxIiAvPgogIDwvZz4KPC9zdmc+Cg==",
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.screenshot_ocr.__name__,
                    description="Taking a screenshot of the VM",
                ),
            ],
            transfer_instructions=SCREENSHOT_OCR_TRANSFER_INSTRUCTIONS,
        )

    def analyse_screenshot_with_llm(self, os: str, ocr_text: str):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ANALYSE_SUPPORT_PROMPT(os, ocr_text),
                    },
                    {
                        "role": "user",
                        "content": f"OS Version: {os}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return None

    def screenshot_ocr(
        self, session_context: dict, context_variables: dict, vm_id: str
    ) -> LlmChatAgentResponse:
        """Call this function if the user is asking you to take a screenshot of a vm and perform OCR on it."""
        try:
            print(f"Screenshot OCR Agent context")
            if not vm_id:
                context_vm_id = get_context_variable(
                    "vm_id", session_context, context_variables
                )
                if not context_vm_id:
                    return LlmChatAgentResponse(
                        status="error",
                        message="No VM ID provided",
                    )
                vm_id = context_vm_id
            ns = ServiceRegistry.get(
                session_context["session_id"],
                NOTIFICATION_SERVICE_NAME,
                NotificationService,
            )
            ls = ServiceRegistry.get(
                session_context["session_id"], LOGGER_SERVICE_NAME, LogService
            )
            ls.info(
                session_context["channel"],
                f"Taking screenshot of vm {vm_id}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Taking a screenshot of  vm",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            data = ServiceRegistry.get(
                session_context["session_id"],
                VM_DATASOURCE_SERVICE_NAME,
                VmDatasourceService,
            )
            vm_details = data.datasource.get_vm(vm_id)
            if not vm_details:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} not found",
                )
            os = vm_details.os
            if not os:
                return LlmChatAgentResponse(
                    status="error",
                    message="No OS provided",
                )
            screenshot = get_vm_screenshot(vm_id)
            if not screenshot.success:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )
            if not screenshot.screenshot:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Performing OCR on the screenshot",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            ocr_service = ServiceRegistry.get(
                session_context["session_id"],
                OCR_SERVICE_NAME,
                OCRService,
            )

            ocr_result = ocr_service.ocr(screenshot.screenshot)
            if not ocr_result.text:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No ocr result provided",
                )
            if ocr_result.average_confidence < 0.5:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="OCR result has an average confidence of less than 50%",
                )
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Analyzing the screenshot",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            analysis = self.analyse_screenshot_with_llm(
                os, "\n".join(ocr_result.strings)
            )
            if not analysis:
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot analysis provided",
                )

            attachment = AttachmentContextVariable(
                name="screenshot",
                id="screenshot",
                type=AttachmentType.IMAGE,
                value=screenshot.screenshot,
            )
            return LlmChatAgentResponse(
                status="success",
                message=analysis,
                actions=[],
                context_variables={"screenshot": attachment},
            )

        except Exception as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to take screenshot of vm {vm_id}, error: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to take screenshot of vm {vm_id}, error: {e}",
                error=str(e),
            )
