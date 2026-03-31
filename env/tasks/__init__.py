from .easy_data_triage import DataQualityTriageTask
from .medium_deployment import DeploymentDecisionTask
from .hard_incident import IncidentCascadeTask
from ..models import TaskID

TASK_REGISTRY: dict = {
    TaskID.DATA_TRIAGE: DataQualityTriageTask,
    TaskID.DEPLOYMENT:  DeploymentDecisionTask,
    TaskID.INCIDENT:    IncidentCascadeTask,
}

__all__ = [
    "DataQualityTriageTask",
    "DeploymentDecisionTask",
    "IncidentCascadeTask",
    "TASK_REGISTRY",
]