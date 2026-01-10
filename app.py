import os
import time
import requests
import warnings
import dspy
from dspy.teleprompt import GEPA

warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

# 1. Ollama Check
def wait_for_ollama():
    url = os.environ.get("OLLAMA_URL")
    if not url:
        print("Fehler: OLLAMA_URL Umgebungsvariable nicht gesetzt.")
        return
    
    print(f"Verbinde mit Ollama unter: {url}...")
    while True:
        try:
            check_url = url.replace("/v1", "") if "/v1" in url else url
            if requests.get(check_url).status_code == 200:
                print("Ollama bereit.")
                break
        except:
            pass
        time.sleep(2)

# 2. LM Setup
wait_for_ollama()

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
    """Schreibe eine Einschlafgesichte mit zwei Jungs als Darsteller. Achte STRENG auf die Formatierung (Absätze und Satzzeichen)."""
    prompt_text = dspy.InputField()
    story = dspy.OutputField()

class DynamicJudgeSignature(dspy.Signature):
    """
    Prüfe den Text extrem präzise auf diese 8 technischen Vorgaben:
    1. DEUTSCH: Ist der Text in deutscher Sprache?
    2. ABSAETZE: Besteht der Text aus exakt 6 durch Leerzeilen getrennte, in etwa gleich lange Blöcke?
    3. HEADER: Beginnt die erste Zeile mit '### '?
    4. FRAGE: Ist das allerletzte Zeichen des gesamten Textes ein Fragezeichen?
	5. CHARACTERS: Sind es genau zwei Brüder als Character(Jungs)?
	6: SLEEP: Ist die Geschichte zum einschlafen geeigent?
	7: HAPPYEND: Hat die Geschichte ein glückliches Ende?
	8: WORDS: Hat der Text zwischen 500 und 600 Wörter?

    
    Antworte für jeden Punkt NUR mit 'Ja' oder 'Nein'.
    Beispiel:
    DEUTSCH: Ja
    ABSAETZE: Nein
    HEADER: Ja
    FRAGE: Nein
	CHRACTERS: Ja
	SLEEP: Ja
	HAPPYEND: Nein
	WORDS: Ja
    """
    text = dspy.InputField()
    assessment = dspy.OutputField(desc="Acht Zeilen mit Ja oder Nein")

# 4. KORRIGIERTE METRIK FÜR GEPA
def story_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    story_content = getattr(pred, 'story', "")
    if not story_content or len(str(story_content)) < 15:
        return 0.0

    with dspy.context(lm=reflection_lm):
        judge = dspy.Predict(DynamicJudgeSignature)
        result = judge(text=story_content)
    
    raw_output = str(result.assessment).lower()
    ja_count = raw_output.count("ja")
    
    return float(min(ja_count, 8) / 8.0)

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
        prompt_text="Zwei Brüder entdecken einen geheimen Dachboden.",
        story=(
            "### Das Geheimnis unterm Dach\n\n"
            "Es war ein regnerischer Dienstag, als die beiden Brüder Lukas und Julian beschlossen, "
            "endlich die verriegelte Tür zum Dachboden zu untersuchen. Lukas, der ältere der beiden, "
            "hielt die Taschenlampe fest umklammert, während der kleine Julian nervös an seinem Ärmel zupfte. "
            "Das alte Holz des Hauses knarrte bei jedem ihrer Schritte, als wollten die Dielen sie warnen. "
            "Doch die Neugier der Jungen war stärker als die Angst vor der Dunkelheit und dem dichten Staub. "
            "Sie schoben den schweren Riegel beiseite und traten in einen Raum, der seit Jahrzehnten "
            "kein Sonnenlicht mehr gesehen hatte. Überall standen alte Truhen und mit Laken bedeckte Möbel.\n\n"
            
            "Im zweiten Abschnitt ihrer Entdeckung fanden sie eine Kiste, die mit goldenen Beschlägen "
            "verziert war. Lukas kniete sich nieder und wischte den Staub beiseite, während Julian "
            "begeistert auf und ab sprang. In der Kiste lagen Briefe, alte Karten und ein Kompass, "
            "der immer noch zuverlässig nach Norden zeigte. Sie fühlten sich wie echte Schatzsucher "
            "in ihrer eigenen Geschichte. Die Luft roch nach altem Papier und getrocknetem Lavendel. "
            "Jedes Objekt in dieser Truhe schien eine eigene Geschichte zu flüstern, die nur darauf "
            "wartete, von den beiden neugierigen Brüdern endlich wieder gehört und verstanden zu werden.\n\n"
            
            "Die Zeit schien stillzustehen, während die Jungen tiefer in die Familiengeschichte eintauchten. "
            "Sie fanden Fotos von ihrem Urgroßvater, der denselben wachen Blick wie Lukas hatte. "
            "Julian lachte leise, als er ein Bild von einem kleinen Hund entdeckte, der genau wie "
            "ihr eigener Welpe aussah. Es war ein friedlicher Moment der Verbundenheit, weit weg "
            "vom Lärm der modernen Welt unter ihnen. Die sanften Geräusche des Regens, der rhythmisch "
            "gegen das Schieferdach trommelte, wirkten wie ein beruhigendes Schlaflied auf die beiden. "
            "Sie fühlten sich hier oben vollkommen sicher und geborgen in ihrem neuen, geheimen Reich.\n\n"
            
            "Lukas begann, eine alte Landkarte zu entfalten, die den Garten ihres Hauses zeigte, "
            "allerdings mit Markierungen, die sie noch nie zuvor gesehen hatten. Julian half ihm, "
            "die Karte glatt zu streichen, und gemeinsam rätselten sie über die Bedeutung der Symbole. "
            "Vielleicht gab es draußen im Garten tatsächlich noch einen echten Schatz zu finden? "
            "Ihre Augen leuchteten vor Begeisterung, doch sie sprachen nur noch im Flüsterton, "
            "um die magische Stille des Ortes nicht zu stören. Die Wärme des Hauses stieg langsam nach oben "
            "und mischte sich mit der kühlen Dachbodenluft, was eine angenehm schläfrige Atmosphäre schuf.\n\n"
            
            "Nachdem sie alles sorgfältig erkundet hatten, beschlossen die Brüder, dass dies ihr neuer "
            "Lieblingsort werden würde. Sie räumten ein paar Decken zusammen und machten es sich in "
            "einer Ecke gemütlich. Es gab keinen Streit, wer die Taschenlampe halten durfte; heute "
            "waren sie ein perfektes Team. Alles fühlte sich richtig und harmonisch an, ein echtes "
            "Abenteuer, das ihre Bindung als Brüder noch weiter stärkte. Zufrieden und mit müden Augen "
            "saßen sie nebeneinander und genossen die Ruhe. Es war das schönste Ende eines langen "
            "Tages, das sie sich hätten vorstellen können, und sie wussten, dass sie heute gut schlafen würden.\n\n"
            
            "Schließlich schlossen sie die schwere Truhe wieder und versprachen sich, am nächsten Tag "
            "wiederzukommen. Die Müdigkeit übermannte sie nun doch, und sie freuten sich auf ihre "
            "warmen Betten und die schönen Träume von Schatzkarten und alten Abenteuern. "
            "Lukas löschte das Licht und sie schlichen leise die Treppe hinunter, beseelt von ihrem "
            "großen Geheimnis. Werden sie morgen den vergrabenen Schatz im Garten wirklich finden?"
        )
    ).with_inputs("prompt_text"),
    
    dspy.Example(
        prompt_text="Zwei Brüder im Wald bei Nacht.",
        story=(
            "### Die Nachtwanderung\n\n"
            "Die beiden Brüder Simon und Marc wagten sich mit ihren Schlafsäcken tief in den herbstlichen Wald. "
            "Um sie herum raschelten die Blätter, und das sanfte Rauschen der Tannen wirkte beruhigend auf ihre Gemüter. "
            "Simon, der Jüngere, hielt die Hand seines großen Bruders fest, während sie einen Platz für ihr Lager suchten. "
            "Der Wald war erfüllt von den friedlichen Geräuschen der Nacht, die wie eine leise Melodie klangen. "
            "Über ihnen funkelten die Sterne klar und hell durch das dichte Blätterdach der alten Eichen. "
            "Es war eine Nacht voller Wunder und die beiden Jungen fühlten sich mutig und bereit für die Ruhe.\n\n"
            
            "In ihrem Lager angekommen, entzündeten sie eine kleine Laterne, die ein warmes, gelbes Licht warf. "
            "Marc erzählte Simon Geschichten von den Tieren des Waldes, die nun ebenfalls in ihren Nestern schliefen. "
            "Die Brüder lachten leise über die Vorstellung, wie ein kleiner Igel sich in sein Laubkissen kuschelte. "
            "Die Luft war frisch und klar, was das Atmen tief und gleichmäßig machte, fast wie von selbst. "
            "Jeder Schatten im Wald wirkte heute Nacht freundlich und einladend, wie ein alter Bekannter. "
            "Die Welt da draußen war weit weg, und hier unter den Bäumen zählte nur der Moment der Stille.\n\n"
            
            "Nach einer Weile legten sie sich in ihre weichen Schlafsäcke und blickten hinauf in die Unendlichkeit. "
            "Simon fragte seinen Bruder nach den Namen der Sternbilder, und Marc erklärte sie ihm mit sanfter Stimme. "
            "Das Wissen, dass sie einander hatten, gab ihnen ein tiefes Gefühl von Sicherheit und Geborgenheit. "
            "Die Jungen spürten, wie ihre Glieder schwer wurden und die wohlige Müdigkeit des Wanderns einsetzte. "
            "Es war eine perfekte Umgebung, um die Sorgen des Tages zu vergessen und einfach nur zu sein. "
            "Die Natur umarmte sie mit ihrer Stille und bereitete sie sanft auf eine lange, erholsame Nacht vor.\n\n"
            
            "Irgendwo in der Ferne rief eine Eule, doch das Geräusch war nicht erschreckend, sondern sehr friedlich. "
            "Marc streichelte seinem Bruder kurz über den Kopf und flüsterte ihm zu, dass alles gut sei. "
            "Sie waren zwei Jungen in einem großen Wald, aber sie fühlten sich kein bisschen einsam oder klein. "
            "Die Verbundenheit zwischen ihnen war wie ein unsichtbares Band, das sie in dieser Nacht schützte. "
            "Alles war harmonisch, und das sanfte Wiegen der Bäume im Wind wirkte wie eine riesige Wiege. "
            "Die Gedanken wurden leiser und die Träume begannen bereits, an die Pforten ihres Bewusstseins zu klopfen.\n\n"
            
            "Ganz allmählich schlossen sich ihre Augen, während der Duft von Kiefernnadeln und Moos sie umgab. "
            "Die Geschichte dieser Nacht war eine von Frieden, Vertrauen und der unendlichen Liebe zwischen Geschwistern. "
            "Kein Unwetter drohte, keine Gefahr lauerte; es war einfach nur die reine, unschuldige Ruhe der Natur. "
            "Sie wussten, dass sie am nächsten Morgen von der warmen Morgensonne geweckt werden würden. "
            "Das glückliche Ende ihrer kleinen Wanderung war die Gewissheit, dass sie gemeinsam alles schaffen konnten. "
            "Mit einem Lächeln auf den Lippen sanken sie tiefer in ihre Kissen und gaben sich dem Schlaf hin.\n\n"
            
            "Der Wald schien nun noch leiser zu werden, als wolle er die beiden Brüder nicht in ihrer Ruhe stören. "
            "Das kleine Licht der Laterne erlosch von selbst, und nur noch das Mondlicht warf silberne Fäden auf den Boden. "
            "Es war eine Nacht, die man niemals vergessen würde, so voller Magie und tiefer, herzlicher Zufriedenheit. "
            "Die Jungen atmeten im Einklang mit dem Wald, ruhig, tief und vollkommen entspannt in ihrer kleinen Welt. "
            "Morgen würde ein neuer Tag voller Entdeckungen beginnen, doch jetzt herrschte nur die sanfte Nacht. "
            "Hörst du auch das leise Atmen der schlafenden Welt?"
        )
    ).with_inputs("prompt_text")
]

# 7. Optimierung
print("Starte GEPA Optimierung mit 5-Arg Metrik...")

optimizer = GEPA(
    metric=story_metric,
    reflection_lm=reflection_lm,
    max_metric_calls=15
)

try:
    optimized_student = optimizer.compile(
        student=StoryStudent(),
        trainset=trainset
    )
    
    print("\n" + "="*60)
    print("OPTIMIERTER PROMPT GEFUNDEN:")
    print(optimized_student.predictor.signature.instructions)
    print("="*60 + "\n")

except Exception as e:
    print(f"Fehler: {e}")
    optimized_student = StoryStudent()

# 8. Testlauf
if __name__ == "__main__":
    test_prompt = "Zwei Brüder beim Drachensteigen am Strand."
    res = optimized_student(prompt_text=test_prompt)
    print(f"GENERIERTE STORY:\n{res.story}")