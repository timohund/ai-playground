import os
import time
import requests
import dspy
from dspy.teleprompt import GEPA
import mlflow

# 1. Ollama Check
while True:
    try:
        if requests.get(os.environ["OLLAMA_URL"]).status_code == 200:
            print("Ollama ready.")
            break
    except:
        pass
    time.sleep(2)

# 2. LM Setup
lm_config = {
    "api_base": os.environ["OLLAMA_URL"],
    "api_key": "ollama",
    "temperature": 0.1,
    "max_tokens": 1000,
}

execution_lm = dspy.LM(model=os.environ["EXECUTION_LLM"], **lm_config)
reflection_lm = dspy.LM(model=os.environ["REFLECTION_LLM"], **lm_config)

dspy.settings.configure(lm=execution_lm)

# 3. Signaturen
class StoryTask(dspy.Signature):
    """Schreibe eine deutsche Geschichte und gib am Ende vier Wahr/Falsch Werte aus."""
    prompt_text = dspy.InputField()
    story = dspy.OutputField(desc="Der Text der Geschichte")
    answers = dspy.OutputField(desc="Vier Wahr/Falsch Werte (True/False)")

class DynamicJudgeSignature(dspy.Signature):
    """
    Bewerte den Text basierend auf folgenden Fragen. 
    Antworte für jede Frage strikt nur mit 'Ja' oder 'Nein'.
    
    Fragen:
    1. Ist der Text auf Deutsch?
    2. Hat der Text exakt vier Absätze?
    3. Ist der Text als Markdown formatiert (benutzt ###)?
    4. Endet der Text mit einer Frage?
    """
    text = dspy.InputField()
    assessment = dspy.OutputField(desc="Eine Liste von Ja/Nein Antworten für jede Frage")

# 4. Metrik (5-Args für GEPA)
def story_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    story_content = getattr(pred, 'story', "")
    if not story_content or len(str(story_content)) < 10:
        return 0.0

    # Liste der Fragen (muss mit der Signature übereinstimmen)
    questions_count = 4

    with dspy.context(lm=reflection_lm):
        judge = dspy.Predict(DynamicJudgeSignature)
        result = judge(text=story_content)
    
    # Wir zählen, wie oft "Ja" in der Antwort vorkommt
    raw_output = str(result.assessment).lower()
    ja_count = raw_output.count("ja")
    
    # Score ist ja_count / Anzahl der Fragen
    return float(min(ja_count, questions_count) / questions_count)

# 5. Modul
class StoryStudent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(StoryTask)

    def forward(self, prompt_text):
        return self.predictor(prompt_text=prompt_text)

# 6. Trainset
trainset = [
    dspy.Example(
        prompt_text="Ein einsamer Wolf im Winter.",
        story="### Der Wolf\n\nEr lief durch den Schnee.\n\nDer Wind war kalt.\n\nEr sah ein Licht.\n\nWo war er?",
        answers="True, True, False, True"
    ).with_inputs("prompt_text"),
    dspy.Example(
        prompt_text="Ein altes Schiff auf dem Meer.",
        story="### Das Schiff\n\nDie Wellen schlugen hoch.\n\nKeiner war an Bord.\n\nGold lag im Deck.\n\nWer segelt hier?",
        answers="True, False, True, True"
    ).with_inputs("prompt_text")
]

# 7. Optimization
mlflow.set_experiment("gepa_dynamic_judge")
with mlflow.start_run():
    optimizer = GEPA(
        metric=story_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=10
    )

    print("Starte GEPA Optimierung mit dynamischem Judge...")
    try:
        optimized_student = optimizer.compile(
            student=StoryStudent(),
            trainset=trainset
        )
    except Exception as e:
        print(f"Fehler: {e}")
        optimized_student = StoryStudent()

# 8. Anzeige des optimierten Prompts und Testlauf
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DER OPTIMIERTE GEPA-PROMPT")
    print("="*60)
    
    # Extraktion der neuen Instruktionen
    best_instructions = optimized_student.predictor.signature.instructions
    print(f"{best_instructions}")
    
    print("="*60 + "\n")

    # Finaler Testlauf
    test_prompt = "Eine Katze, die den Mond fangen will."
    res = optimized_student(prompt_text=test_prompt)
    
    print(f"GENERIERTE STORY:\n{res.story}")
    print(f"\nANTWORTEN (True/False): {res.answers}")