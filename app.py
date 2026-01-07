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

# 2. LM Setup (Wichtig: Wir schalten den automatischen JSON-Adapter aus)
# Wir nutzen eine höhere Repeat-Penalty, um die "Wald und Hügel"-Schleifen zu stoppen.
lm_config = {
    "api_base": os.environ["OLLAMA_URL"],
    "api_key": "ollama",
    "temperature": 0.1,
    "max_tokens": 300,
}
execution_lm = dspy.LM(model="ollama/tinyllama", **lm_config)
reflection_lm = dspy.LM(model="ollama/tinyllama", **lm_config)

# Wir setzen den Adapter auf None, um reines Text-Processing zu machen
dspy.settings.configure(lm=execution_lm, adapter=None)

# 3. Robuste Signature
class StoryTask(dspy.Signature):
    """
    Aufgabe: Schreibe eine kurze deutsche Geschichte.
    Gib am Ende exakt vier Zeilen mit True oder False aus.
    Format:
    Story: [Text]
    Answers: [True/False, True/False, True/False, True/False]
    """
    prompt_text = dspy.InputField()
    story = dspy.OutputField(desc="Der Text der Geschichte")
    answers = dspy.OutputField(desc="Vier Wahr/Falsch Werte durch Komma getrennt")

# 4. Modul
class StoryStudent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(StoryTask)

    def forward(self, **kwargs):
        # Wir rufen den Predictor auf. Ohne Adapter gibt er das zurück, was er parsen kann.
        return self.predictor(**kwargs)

# 5. Metrik (GEPA 5-args) - Extrem tolerant gegenüber TinyLlama-Fehlern
def story_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = 0.0
    try:
        # Extrahiere Text aus dem 'answers' Feld oder suche im Gesamtstring danach
        ans_text = str(getattr(pred, 'answers', "")).lower()
        
        # Zähle wie oft True vorkommt (einfachster Check für schwache Modelle)
        trues = ans_text.count("true")
        score = min(trues, 4) 
        
        return float(score / 4.0)
    except:
        return 0.0

# 6. Trainset (Mit Dummy-Werten für GEPA)
trainset = [
    dspy.Example(
        prompt_text="Write a short story in German about a mountain.",
        story="Ein hoher Berg.", 
        answers="True, True, True, True"
    ).with_inputs("prompt_text")
]

# 7. Optimization
mlflow.set_experiment("gepa_no_adapter_fix")
with mlflow.start_run():
    # GEPA braucht eine Reflection-LM, die Instruktionen ändert.
    # Da TinyLlama schwach ist, reduzieren wir die Versuche.
    optimizer = GEPA(
        metric=story_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=5 
    )

    print("Starte GEPA Optimierung (Manual Mode)...")
    try:
        # Wir kompilieren das Modul
        optimized_student = optimizer.compile(
            student=StoryStudent(),
            trainset=trainset
        )
    except Exception as e:
        print(f"GEPA Fehler (übersprungen): {e}")
        optimized_student = StoryStudent()

# 8. Anzeige des optimierten Prompts
print("\n" + "="*50)
print("GEPA OPTIMIERTER PROMPT")
print("="*50)
try:
    print(optimized_student.predictor.signature.instructions)
except:
    print("Keine Instruktionen gefunden.")
print("="*50 + "\n")

# 9. Testlauf
if __name__ == "__main__":
    print("Generiere finale Geschichte...")
    res = optimized_student(prompt_text="Schreibe über eine kleine Katze.")
    print(f"STORY:\n{getattr(res, 'story', 'Kein Text generiert')}")
    print(f"ANSWERS: {getattr(res, 'answers', 'Keine Antworten')}")
