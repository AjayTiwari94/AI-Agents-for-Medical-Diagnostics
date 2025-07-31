from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


PROMPTS = {
    "cardiologist": """
        Act as a cardiologist. You will receive a medical report of a patient.
        Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
        Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient’s symptoms. Rule out any underlying heart conditions, such as arrhythmias or structural abnormalities.
        Output: Provide a concise summary of your findings and recommended next steps.
        Medical Report: {medical_report}
    """,
    "psychologist": """
        Act as a psychologist. You will receive a patient's report.
        Task: Review the patient's report and provide a psychological assessment.
        Focus: Identify any potential mental health issues, such as anxiety, depression, or trauma, that may be affecting the patient's well-being.
        Output: Provide a concise summary of your findings and recommended next steps.
        Patient's Report: {medical_report}
    """,
    "pulmonologist": """
        Act like a pulmonologist. You will receive a patient's report.
        Task: Review the patient's report and provide a pulmonary assessment.
        Focus: Identify any potential respiratory issues, such as asthma, COPD, or lung infections, that may be affecting the patient's breathing.
        Output: Provide a concise summary of your findings and recommended next steps.
        Patient's Report: {medical_report}
    """,
    "multidisciplinary_team": """
        Act like a multidisciplinary team of healthcare professionals.
        You will receive reports from a Cardiologist, Psychologist, and Pulmonologist.
        Task: Review the specialist reports, analyze them, and come up with a list of 3 possible health issues for the patient.
        For each issue, provide a clear rationale based on the provided reports. One issue must be marked as the primary diagnosis.

        Cardiologist Report: {cardiologist_report}
        Psychologist Report: {psychologist_report}
        Pulmonologist Report: {pulmonologist_report}
        
        {format_instructions}
    """
}

class HealthIssue(BaseModel):
    """A data model for a single health diagnosis."""
    diagnosis: str = Field(description="The name of the possible health diagnosis")
    rationale: str = Field(description="The rationale for why this is a possible diagnosis, citing the specialist reports")
    is_primary: bool = Field(description="Set to true if this is the most likely primary diagnosis, otherwise false")

class FinalAnalysis(BaseModel):
    """A data model for the complete final analysis from the team."""
    analysis: list[HealthIssue] = Field(description="A list containing the primary diagnosis and two differential diagnoses")

# --- Agent Classes ---
class SpecialistAgent:
    """An agent representing a single medical specialist."""
    def __init__(self, role: str, model_name: str = "gpt-4o"):
        self.role = role
        self.model = ChatOpenAI(temperature=0, model=model_name)
        prompt_text = PROMPTS[self.role.lower()]
        self.prompt_template = PromptTemplate.from_template(prompt_text)
        self.chain = self.prompt_template | self.model

    def run(self, medical_report: str) -> str:
        """Runs the agent's analysis. Data is passed here, not in the constructor."""
        return self.chain.invoke({"medical_report": medical_report}).content

class MultidisciplinaryTeam:
    """An agent that synthesizes reports from multiple specialists."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = ChatOpenAI(temperature=0, model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
        prompt_text = PROMPTS["multidisciplinary_team"]
        self.prompt_template = PromptTemplate.from_template(
            template=prompt_text,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt_template | self.model | self.parser

    def run(self, cardiologist_report: str, psychologist_report: str, pulmonologist_report: str) -> FinalAnalysis:
        """Runs the synthesis process on the collected specialist reports."""
        return self.chain.invoke({
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        })
