from govform_env.env import GovFormEnv
from govform_env.models import GovFormAction
from govform_env.graders.task1_aadhaar import AadhaarGrader
from govform_env.graders.task2_income import IncomeGrader
from govform_env.graders.task3_passport import PassportGrader

def test_scenario(task_id, actions):
    print(f"\n--- Testing Task: {task_id} ---")
    env = GovFormEnv(task_id)
    env.reset()
    
    # Choose the right grader
    if task_id == "aadhaar_update": grader = AadhaarGrader()
    elif task_id == "income_certificate": grader = IncomeGrader()
    else: grader = PassportGrader()
    
    # Perform custom actions
    for field, val in actions.items():
        env.step(GovFormAction(field_name=field, value=val))
        print(f"Action: {field} -> {val}")
    
    # Check the final grade
    score = grader.grade(env.state())
    print(f"FINAL SCORE: {score:.3f}")

# EXAMPLES TO TRY:

# 1. Testing a partially filled Aadhaar Form (Expected: 0.50)
test_scenario("aadhaar_update", {
    "full_name": "Arjun Das",
    "aadhaar_number": "123456789012",
    "state": "West Bengal"
})

# 2. Testing an Invalid Income Certificate (Expected: Low Score)
test_scenario("income_certificate", {
    "certificate_type": "BPL",
    "annual_income": "900000" # Too high for BPL!
})
