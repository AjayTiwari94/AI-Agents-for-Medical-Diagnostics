import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import SpecialistAgent, MultidisciplinaryTeam

def setup_logging(output_dir: str):
    """Configures logging to output to both the console and a file."""
    log_dir = os.path.join(output_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "analysis.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def main(report_path: str, output_dir: str):
    """Controls the full workflow from loading data to saving the final report."""
    
    log_file = setup_logging(output_dir)
    
    # --- 1. Setup and Input Validation ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("FATAL: OPENAI_API_KEY not found in .env file. Please create one.")
        sys.exit(1)

    try:
        with open(report_path, "r") as file:
            medical_report = file.read()
        logging.info(f"Successfully loaded report: {os.path.basename(report_path)}")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at '{report_path}'")
        sys.exit(1)

    # --- 2. Initialize and Run Specialist Agents Concurrently ---
    specialist_roles = ["Cardiologist", "Psychologist", "Pulmonologist"]
    agents = {role: SpecialistAgent(role=role) for role in specialist_roles}
    responses = {}

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        future_to_role = {
            executor.submit(agent.run, medical_report): role 
            for role, agent in agents.items()
        }

        logging.info(f"Dispatching tasks to {len(specialist_roles)} specialist agents...")
        for future in as_completed(future_to_role):
            role = future_to_role[future]
            try:
                report = future.result()
                responses[role] = report
                logging.info(f"✓ Report received from {role}")
            except Exception as e:
                logging.error(f"✗ {role} agent failed: {e}")
                responses[role] = f"AGENT FAILED TO PRODUCE REPORT. REASON: {e}"

    # --- 3. Run Synthesis Agent and Generate Final Report ---
    if len(responses) < len(specialist_roles):
        logging.critical("Could not proceed with final analysis as one or more specialist agents failed.")
        sys.exit(1)

    logging.info("All specialist reports received. Running Multidisciplinary Team...")
    try:
        team_agent = MultidisciplinaryTeam()
        final_analysis = team_agent.run(
            cardiologist_report=responses.get("Cardiologist", "N/A"),
            psychologist_report=responses.get("Psychologist", "N/A"),
            pulmonologist_report=responses.get("Pulmonologist", "N/A")
        )
    except Exception as e:
        logging.error(f"FATAL: The Multidisciplinary Team agent failed: {e}")
        sys.exit(1)

    # --- 4. Format and Save the Output ---
    report_text = "### Final Multidisciplinary Team Analysis ###\n\n"
    for issue in sorted(final_analysis.analysis, key=lambda x: not x.is_primary):
        tag = "Primary Diagnosis" if issue.is_primary else "Differential Diagnosis"
        report_text += f"**{tag}: {issue.diagnosis}**\n"
        report_text += f"   - **Rationale:** {issue.rationale}\n\n"

    output_filename = f"analysis_{os.path.basename(report_path)}"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as file:
        file.write(report_text)
    
    logging.info("-" * 50)
    logging.info(f"🎉 Analysis complete! Final report saved to: {output_path}")
    logging.info(f"📄 A detailed log of this run was saved to: {log_file}")
    logging.info("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a multi-agent AI analysis on a medical report.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example usage:\npython main.py \"./Medical Reports/report.txt\" -o \"./results\""
    )
    parser.add_argument(
        "report_path",
        type=str,
        help="The full path to the patient's medical report text file."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./results",
        help="The directory to save the output report and log file (default: ./results)."
    )
    args = parser.parse_args()
    main(args.report_path, args.output_dir)
