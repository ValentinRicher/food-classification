from azureml.core.webservice import Webservice
from azureml.core import Workspace

from src.settings.settings import azure_config

workspace_name = azure_config["workspace_name"]
azure_subscription_id = azure_config["subscription_id"]
resource_group = azure_config["resource_group"]
workspace = workspace = Workspace.get(
    name=workspace_name,
    subscription_id=azure_subscription_id,
    resource_group=resource_group,
)

aks_service = Webservice(workspace, "food-service-2")
aks_service.update(enable_app_insights=True)
