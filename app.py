import os
import time
import requests
import warnings
import dspy
import re
import litellm
import mlflow
from dspy.teleprompt import GEPA

# --- GLOBALE VARIABLEN ---
best_score_so_far = -1.0

# --- STABILITÄTS-SETUP ---
warnings.filterwarnings("ignore")
# Timeout auf 25 Minuten erhöhen für lokale Hardware-Generierung
litellm.request_timeout = 1500 

# --- MLFLOW SETUP ---
# Nutzt die Namen aus deinem Docker-Compose File
ml_host = os.environ.get('MLFLOW_HOST', 'mlflow-server')
ml_port = os.environ.get('MLFLOW_PORT', '5000')
mlflow.set_tracking_uri(f"http://{ml_host}:{ml_port}")
mlflow.set_experiment("Story_Optimization_GEPA")

# --- 1. ZENTRALE REGEL-DEFINITION ---
STORY_CONSTRAINTS = """
0. SPRACHE: Alle Anweisungen und der Text MÜSSEN auf DEUTSCH sein.
1. HEADER: Starte die erste Zeile IMMER mit '### ' und einem kreativen Titel.
2. ABSÄTZE: Erzeuge EXAKT 6 Blöcke (Absätze). Trenne diese zwingend durch eine Leerzeile.
3. LÄNGE: Schreibe SEHR ausführlich. Ziel sind 750 Wörter. (Minimum 600, Maximum 800 Wörter).
4. CHARAKTERE: Hauptpersonen sind genau zwei Brüder (Jungs).
5. TEAMPLAY: Die Brüder lösen ein Problem nur durch Zusammenarbeit.
6. STIMMUNG: Ruhige, friedliche Einschlaf-Atmosphäre.
7. ABSCHLUSS: Das allerletzte Zeichen des gesamten Textes MUSS ein Fragezeichen sein.
8. NUR STORY: Gib NUR die Geschichte aus. Kein 'Hier ist deine Geschichte', keine Kommentare.
9. HAPPY END: Die Geschichte endet immer positiv und geborgen.
10. FANTASY: Baue sanfte magische Elemente ein.
"""

# --- 2. SIGNATUREN ---
class StoryTask(dspy.Signature):
    __doc__ = f"Schreibe eine extrem ausführliche Einschlafgeschichte auf Deutsch unter Einhaltung dieser Regeln:\n{STORY_CONSTRAINTS}"
    prompt_text = dspy.InputField(desc="Thema/Handlung der Geschichte.")
    story = dspy.OutputField(desc="### Titel\n\n[6 sehr lange Absätze]\n\nLetztes Zeichen: ?")

class DynamicJudgeSignature(dspy.Signature):
    __doc__ = f"Prüfe den Text extrem präzise auf die Einhaltung der 11 technischen Vorgaben (Ja/Nein):\n{STORY_CONSTRAINTS}"
    text = dspy.InputField()
    assessment = dspy.OutputField(desc="11 Zeilen mit Ja oder Nein.")

# --- 3. INFRASTRUKTUR & LM SETUP ---
def wait_for_ollama():
    url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    print(f"Verbinde mit Ollama unter: {url}...")
    while True:
        try:
            if requests.get(url.replace("/v1", "")).status_code == 200:
                print("Ollama bereit."); break
        except: pass
        time.sleep(2)

wait_for_ollama()

lm_config = {"api_base": os.environ["OLLAMA_URL"], "api_key": "ollama", "max_tokens": 4000}
execution_lm = dspy.LM(model=os.environ["EXECUTION_LLM"], temperature=0.6, **lm_config)
reflection_lm = dspy.LM(model=os.environ["REFLECTION_LLM"], temperature=0.7, **lm_config)
dspy.settings.configure(lm=execution_lm)

# --- 4. METRIK (Inkl. Best Prompt Tracker & MLflow) ---
def story_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    global best_score_so_far
    story_content = getattr(pred, 'story', "")
    if not story_content or len(str(story_content)) < 150: return 0.0
    
    with dspy.context(lm=reflection_lm):
        judge = dspy.Predict(DynamicJudgeSignature)
        result = judge(text=story_content)
    
    raw_output = str(result.assessment).lower()
    ja_count = raw_output.count("ja")
    words = len(re.findall(r'\w+', story_content))
    
    # Wortzahl-Logik
    word_score = 1.0 if 600 <= words <= 850 else (0.4 if words > 400 else 0.1)
    # Absatz-Logik
    paragraphs = [p for p in story_content.split('\n\n') if len(p.strip()) > 30]
    p_score = 1.0 if len(paragraphs) == 6 else 0.5
    
    final_score = float((ja_count / 11.0) * word_score * p_score)
    
    if mlflow.active_run():
        mlflow.log_metric("current_score", final_score)
        mlflow.log_metric("word_count", words)

    if final_score > best_score_so_far:
        best_score_so_far = final_score
        with open("best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(f"--- Highscore: {final_score} ---\n")
            f.write(f"Wortzahl: {words} | Ja-Punkte: {ja_count}\n")
            f.write("-" * 30 + "\n")
            if trace is not None:
                f.write(str(trace[0][0].predictor.signature.instructions))
            else:
                f.write("Initialer Lauf")
        print(f" >>> Neuer Bestwert: {final_score}! In best_prompt.txt gesichert.")

    return final_score

# --- 5. MODUL ---
class StoryStudent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(StoryTask)
    def forward(self, prompt_text):
        return self.predictor(prompt_text=prompt_text)

# --- 6. DATENSATZ (GOLDSTANDARD EXAMPLES) ---
all_examples = [
    dspy.Example(
        prompt_text="Lukas und Finn in der Kristallhöhle.",
        story="""### Die Kristallhöhle der Sternenläufer

Lukas und sein kleiner Bruder Finn saßen oft am Fenster ihres gemeinsamen Zimmers und beobachteten, wie der Mond die alten Eichen im Garten in silbrig-blaues Licht tauchte. Heute war ein besonderer Abend, denn sie hatten im Keller eine Karte gefunden, die den Weg zu einer geheimen Kristallhöhle beschrieb, die angeblich nur im Traum existierte. Die beiden Jungs, die seit ihrer Geburt unzertrennlich waren, rückten ihre Kissen zurecht und hielten sich an den Händen. Sie wussten, dass sie diese Reise nur gemeinsam antreten konnten, da ihre Kräfte sich erst im Teamplay voll entfalteten. Lukas war der mutige Pfadfinder, während Finn die Gabe besaß, das Flüstern der Sterne zu verstehen. Gemeinsam schlossen sie die Augen und spürten, wie das Bett sanft wie eine Wolke in die Höhe stieg.

Der Weg führte sie durch einen Wald aus schwebenden Laternenblumen, deren Duft nach Lavendel und warmer Milch die Luft erfüllte. Lukas ging voran und schob die leuchtenden Farne beiseite, damit sein kleiner Bruder Finn nicht stolperte. Sie sprachen kaum, denn ihre Verbindung war so tief, dass ein Blick genügte, um den nächsten Schritt zu planen. Die Jungs erreichten einen tiefen Abgrund, über den keine Brücke führte, doch sie gerieten nicht in Panik. Finn flüsterte ein altes Lied der Nachtigallen, und plötzlich begannen sich die Lichtstrahlen der Sterne zu einer gläsernen Brücke zu verweben. Nur weil sie als Team agierten und einander blind vertrauten, konnten sie den glitzernden Pfad sicher überqueren. Die Stille des Waldes wirkte beruhigend auf ihren Atem, während sie tiefer in die Welt der Träume vordrangen.

Vor dem Eingang der Kristallhöhle wartete ein riesiger, sanfter Bär aus purem Nebel, der den Weg versperrte. Er verlangte ein Rätsel zu lösen, das nur gelöst werden konnte, wenn zwei Herzen im gleichen Takt schlugen. Lukas legte seinen Arm um Finns Schultern, und gemeinsam dachten sie an ihre schönsten Erinnerungen: das gemeinsame Baumhausbauen und die geheimen Mitternachtssnacks. Der Nebelbär lächelte mit leuchtenden Augen und trat zur Seite, denn die reine Geschwisterliebe der zwei Jungs war der Schlüssel zum Tor. Die Höhle im Inneren glühte in sanften Pastelltönen, die sich wie weiche Decken um ihre Gedanken legten. Überall hingen Kristalle, die leise im Wind des Untergrunds klangen wie sanfte Windspiele, die jedes Kind sofort in einen tiefen Schlummer wiegen würden.

Im Zentrum der Höhle fanden sie den großen Wunschstein, der die Kraft besaß, Träume wahr werden zu lassen. Doch statt sich Spielzeug oder Süßigkeiten zu wünschen, sahen sich Lukas und Finn an und nickten gleichzeitig. Ihr Wunsch war es, dass jedes Kind auf der Welt in dieser Nacht einen friedlichen Schlaf finden und von fliegenden Inseln träumen könne. In diesem Moment verschmolzen ihre Schatten an der Höhlenwand zu einer Einheit, und ein warmer Lichtstrom breitete sich über den Horizont aus. Das Teamplay der zwei Brüder hatte eine Magie entfacht, die weit über ihre eigene Vorstellungskraft hinausging. Die Kristalle begannen heller zu leuchten, aber ihr Licht war nicht blendend, sondern so beruhigend wie das Glimmen einer kleinen Kerze im dunklen Flur.

Als das Ziel ihrer Reise erreicht war, spürten die Jungs eine angenehme Schwere in ihren Gliedern, die sie sanft zurück in Richtung ihrer Betten zog. Das Abenteuer neigte sich dem Ende zu, und der Erfolg ihrer Mission erfüllte sie mit tiefer Zufriedenheit und einem glücklichen Lächeln. Sie wussten, dass sie am nächsten Morgen gemeinsam aufwachen würden, bereit für neue Taten im echten Leben. Die Welt der Fantasie verneigte sich vor den zwei Brüdern, die bewiesen hatten, dass Mut und Zusammenhalt selbst die Sterne bewegen konnten. Die weichen Kissen fingen sie wieder auf, und die Decke fühlte sich nun an wie die warmen Flügel eines Schutzengels. Alles war friedlich, und die Ruhe der Nacht legte sich wie ein schützender Kokon um das ganze Haus.

Nun liegen die beiden Brüder Lukas und Finn sicher in ihren Federn und lassen die Bilder der Kristallhöhle in ihren Gedanken weitertanzen. Das sanfte Rauschen des Windes draußen vor dem Fenster klingt wie ein fernes Schlaflied, das nur für sie beide gesungen wird. Ihre Abenteuerlust ist für heute gestillt, und die Gewissheit, dass sie sich immer aufeinander verlassen können, ist das schönste Happy End, das man sich vorstellen kann. Die Sterne am Himmel funkeln nun ein wenig heller, als wollten sie den zwei Helden für ihre Güte danken. Die Augenlider sind nun ganz schwer, und der Atem geht ruhig und gleichmäßig im Takt der stillen Nacht. Kannst du auch schon die leisen Klänge der Kristalle hören und bist du bereit für deine eigene Reise in das Land der Träume?"""
    ).with_inputs("prompt_text"),
    dspy.Example(
        prompt_text="Julian und Marc reisen zum Wolkenschloss.",
        story="""### Die geheime Reise zum Wolkenschloss

Julian und sein älterer Bruder Marc hatten sich eine Festung aus Decken gebaut, die sich heute Nacht in ein echtes Luftschiff verwandelte. Mit leisen Sohlen schlichen sie durch ihre Fantasie, bis der Teppich unter ihnen abhob und sie sanft durch das offene Fenster in den sternenklaren Nachthimmel trug. Die beiden Jungs hielten sich an den Rändern ihrer fliegenden Decke fest und spürten den kühlen, aber angenehmen Nachtwind in ihren Haaren. Marc, der immer einen kühlen Kopf bewahrte, übernahm das Steuer, während Julian mit seinem Fernrohr aus Pappe nach den leuchtenden Wegweisern suchte. Es war ihr Ziel, das Wolkenschloss zu finden, in dem die Träume für den nächsten Tag geschmiedet wurden, und nur als eingespieltes Team konnten sie die tückischen Luftströmungen überwinden.

Der Himmel war erfüllt von fliegenden Fischen, die in den schillerndsten Farben des Regenbogens leuchteten und die beiden Jungs neugierig begleiteten. Julian lachte leise, als ein besonders kleiner Fisch seine Nase kitzelte, während Marc sich darauf konzentrierte, die Decke sicher an den schlafenden Riesenwolken vorbeizusteuern. Sie mussten eng zusammenarbeiten, denn immer wenn der Wind drehte, mussten sie ihr Gewicht gleichzeitig verlagern, um nicht vom Kurs abzukommen. Das Teamplay der zwei Brüder war bewundernswert, da sie ohne Worte verstanden, was der andere gerade brauchte. Unter ihnen glitzerten die Lichter der Stadt wie kleine Diamanten, die in ein samtenes Tuch eingestickt waren, und die Welt wirkte so friedlich und still, dass man das Herz der Erde schlagen hören konnte.

Plötzlich tauchte vor ihnen ein dichtes Feld aus Zuckerwatte-Nebel auf, der den Weg zum Schloss vollkommen verschleierte. Julian erinnerte sich an eine Geschichte, die ihre Mutter ihnen oft vorgelesen hatte, und wusste, dass man diesen Nebel nur mit purer Freude vertreiben konnte. Er begann leise zu summen, und Marc stimmte mit einer tiefen, beruhigenden Stimme in das Lied ein, bis der Nebel vor ihrer gemeinsamen Fröhlichkeit zurückwich. Die Fantasie der Jungs kannte keine Grenzen, und mit jedem Ton des Liedes wurde der Weg klarer und heller. Sie sahen nun die hohen Türme des Wolkenschlosses, die aus gefrorenem Mondlicht bestanden und sanft in der Dunkelheit schimmerten. Das Schloss war ein Ort der absoluten Ruhe, an dem alle Sorgen wie kleine Seifenblasen zerplatzten und verschwanden.

Am Tor des Schlosses wurden sie von einem Wächter aus Sternenstaub empfangen, der ihnen den Zutritt nur gewährte, wenn sie bewiesen, dass sie wahre Helden des Alltags waren. Marc erzählte von der Zeit, als Julian hingefallen war und er ihn getröstet hatte, und Julian berichtete, wie Marc ihm immer bei den Hausaufgaben half. Der Wächter war tief beeindruckt von der Loyalität der zwei Brüder und öffnete die schweren Tore mit einem silbernen Schlüssel. Im Inneren des Schlosses roch es nach frisch gebackenen Plätzchen und warmer Vanille, ein Duft, der sofort jedes Kind zum Gähnen brachte. Die Jungs durften sich auf die weichsten Wolken setzen, die jemals existiert hatten, und spürten, wie alle Anspannung von ihren kleinen Körpern abfiel.

Es war Zeit, den Rückweg anzutreten, denn der Sandmann wartete bereits mit seinem goldenen Säckchen am Horizont. Die Rückreise war ruhig und sicher, da die Jungs nun wussten, dass sie gemeinsam jede Hürde meistern konnten, egal wie hoch sie am Himmel schwebte. Sie landeten sanft wieder in ihrem Zimmer, und die Deckenfestung schmiegte sich wieder eng um sie, als wären sie nie fortgewesen. Das Abenteuer hatte sie noch enger zusammengeschweißt, und das glückliche Ende ihrer Reise war in ihren entspanzten Gesichtern deutlich zu erkennen. Die Fantasie hatte ihnen einen Weg gewiesen, den sie niemals vergessen würden, und die Wärme ihres Zusammenhalts schützte sie vor der Kühle der Nacht. Alles im Raum war nun in ein sanftes Gold gehüllt, das vom verblassten Sternenstaub an ihrer Kleidung stammte.

Jetzt liegen Marc und Julian tief in ihre Kissen gekuschelt und lassen die Erlebnisse der Nacht wie einen schönen Film vor ihren inneren Augen ablaufen. Ihr Atem ist schwer und ruhig geworden, während die Schatten an der Wand friedliche Muster zeichnen, die zum Träumen einladen. Die Geschichte der Wolkensegler ist nun zu Ende, aber in ihren Herzen brennt das Licht des Abenteuers ewig weiter. Es ist die perfekte Zeit, um die Augen zu schließen und sich von der Sanftheit der Nacht davontragen zu lassen, genau wie die beiden Helden es getan haben. Die Stille im Haus ist vollkommen, und nur das Ticken der Uhr im Flur begleitet sie in den tiefen, erholsamen Schlaf. Spürst du auch schon, wie die Müdigkeit dich wie eine weiche Wolke umarmt und bist du bereit für dein eigenes Schloss?"""
    ).with_inputs("prompt_text"),
    dspy.Example(
        prompt_text="Elias und Noah finden eine singende Meeresmuschel.",
        story="""### Das Geheimnis der singenden Meeresmuschel

Elias und sein kleiner Bruder Noah verbrachten ihre Sommerferien am liebsten am alten Leuchtturm, wo die Wellen Geschichten von versunkenen Städten erzählten. Eines Abends, als die Sonne wie eine reife Orange im Ozean versank, fanden die beiden Jungs eine Muschel, die in allen Farben des Regenbogens schimmerte und leise vibrierte. Noah legte sie an sein Ohr und hörte eine Melodie, die so sanft war wie das Rauschen des Windes in den Dünen. Die Jungs wussten sofort, dass dies keine gewöhnliche Muschel war, sondern ein Wegweiser zu einem verborgenen Unterwassergarten. Mit mutigen Herzen und Schnorcheln bewaffnet, wateten sie ins warme Wasser, das sich anfühlte wie eine flüssige Umarmung der Natur. Es war ihr erstes großes Teamplay-Abenteuer unter der Meeresoberfläche, und sie hielten sich fest an den Händen, um gemeinsam abzutauchen.

Unter Wasser verwandelte sich die Welt in ein leuchtendes Wunderreich aus Korallen, die wie bunte Bäume im Rhythmus der Strömung tanzten. Elias wies den Weg, während Noah mit einer kleinen Taschenlampe die dunklen Spalten beleuchtete, um sicherzustellen, dass sie keine versteckten Schätze übersahen. Die beiden Jungs kommunizierten durch Handzeichen, die sie sich im Laufe der Jahre selbst beigebracht hatten, und funktionierten wie eine perfekte Einheit. Sie begegneten einer Gruppe von Meeresschildkröten, die sie freundlich begleiteten und ihnen zeigten, wie man sich mühelos durch die Fluten gleiten lässt. Die Stille des Ozeans wirkte beruhigend auf ihre Gedanken, und das sanfte Blau des Wassers hüllte sie in eine schützende Decke aus purer Fantasie. Überall glitzerten kleine Luftblasen wie Perlen, die langsam zur Oberfläche aufstiegen und das Mondlicht einfingen.

In der Mitte des Gartens stand eine alte Statue aus Sandstein, die eine riesige Perle hielt, welche das Licht des Mondes speicherte. Die Muschel in Noahs Hand begann lauter zu singen, als sie sich der Statue näherten, und die Melodie füllte den gesamten Raum zwischen den Korallen. Elias erkannte, dass die Statue ein Teil eines Puzzles war, das nur gelöst werden konnte, wenn zwei Menschen gleichzeitig an den Seiten drückten. Mit vereinten Kräften und vollem Vertrauen ineinander aktivierten die zwei Brüder den Mechanismus, und ein sanfter Lichtstrom breitete sich aus. Es war ein Moment des puren Glücks, als sie sahen, wie der Garten in ein Paradies aus leuchtenden Pflanzen verwandelte, die wie Nachtlichter strahlten. Das glückliche Ende ihrer Suche war nah, denn sie hatten bewiesen, dass sie gemeinsam alles erreichen konnten.

Nachdem sie das Geheimnis der Muschel gelüftet hatten, fühlten sich die Jungs angenehm müde, als hätte das Wasser ihre Glieder sanft massiert. Sie schwammen langsam zurück zum Strand, wobei sie sich gegenseitig stützten, wenn eine Welle einmal etwas kräftiger gegen sie drückte. Die Rückkehr ans Ufer war wie ein sanftes Erwachen aus einem wunderschönen Traum, während der Sand unter ihren Füßen noch warm vom vergangenen Tag war. Sie legten sich kurz in den weichen Dünensand und blickten hinauf zu den Sternen, die heute besonders hell für sie zu leuchten schienen. Ihre Fantasie war reich beschenkt worden, und das Gefühl der Geborgenheit hüllte sie ein wie eine dicke Wolldecke. Die Nacht war still geworden, und nur das ferne Rauschen der Brandung sang ihnen ein privates Schlaflied.

Wieder zu Hause angekommen, schlichen sie leise in ihr Zimmer, das nach Lavendel und frischer Wäsche duftete, und kuschelten sich in ihre Betten. Elias deckte seinen kleinen Bruder Noah liebevoll zu, und sie flüsterten sich noch einmal die Highlights ihres Abenteuers zu, bis ihre Stimmen immer leiser wurden. Das Teamplay des Tages hatte sie stolz gemacht und ihnen gezeigt, wie wichtig es ist, jemanden an seiner Seite zu haben. Die Bilder der leuchtenden Korallen tanzten noch hinter ihren geschlossenen Augenlidern, während die Schwere des Schlafes sie nun endgültig einholte. Alles war friedlich, sicher und voller Liebe in diesem kleinen Raum am Ende der Welt. Die Gewissheit, dass morgen ein neuer Tag voller Wunder auf sie wartete, ließ sie mit einem Lächeln auf den Lippen einschlafen.

Nun schlafen Elias und Noah tief und fest, während der Mond schützend über ihr Haus wacht und die Schatten der Nacht friedlich draußen bleiben. Ihre Reise in den Unterwassergarten ist nun eine kostbare Erinnerung, die sie in ihren Träumen weiter begleiten wird, bis die Sonne sie wieder weckt. Die Welt da draußen ist zur Ruhe gekommen, und jedes Tier und jede Pflanze scheint den Atem anzuhalten, um den Schlaf der Gerechten nicht zu stören. Es ist die Zeit der Stille, in der nur das Herz leise klopft und die Fantasie ihre buntesten Blüten treibt. Die Kissen sind weich, die Träume sind süß, und die Geborgenheit der Familie ist das stärkste Band, das es gibt. Kannst du auch die sanfte Melodie der Meeresmuschel in deinem Herzen hören und bist du bereit für dein eigenes blaues Abenteuer?"""
    ).with_inputs("prompt_text"),
    dspy.Example(
        prompt_text="Leo und Max steigen in einen magischen Spielzeugzug.",
        story="""### Die Fahrt mit dem magischen Spielzeugzug

In der Mitte des Kinderzimmers von Leo und seinem Bruder Max stand eine hölzerne Eisenbahn, die heute Nacht ganz besonders hell im Mondschein glänzte. Als die Turmuhr zwölfmal schlug, begannen die kleinen Waggons zu wachsen, bis sie groß genug waren, dass die zwei Jungs bequem darin Platz nehmen konnten. Leo, der ältere der beiden, setzte sich mutig auf den Platz des Lokführers, während Max mit einer goldenen Trillerpfeife das Signal zur Abfahrt gab. Der Zug setzte sich mit einem sanften Schnaufen in Bewegung und verließ die Schienen auf dem Boden, um direkt auf einem Pfad aus Sternenstaub durch das Zimmer zu gleiten. Die beiden Jungs lachten vor Vergnügen, als sie sahen, wie ihre Kuscheltiere ihnen vom Bettrand aus freundlich zuwinkten. Es war der Beginn einer Reise in das Land der vergessenen Spielzeuge, die sie nur durch enges Teamplay bestehen konnten.

Der Zug sauste durch ein Gebirge aus weichen Kissen und überquerte Brücken, die aus bunten Bauklötzen gebaut waren und bei jeder Berührung leise Musik machten. Max half Leo dabei, die Kohlen aus Schokoladenstücken in den Ofen der Lokomotive zu schaufeln, damit der Dampf nach warmer Milch und Honig duftete. Sie arbeiteten Hand in Hand, denn nur wenn das Feuer gleichmäßig brannte, konnte der Zug seine magische Geschwindigkeit halten und sie sicher ans Ziel bringen. Die Fantasie der Jungs ließ immer neue Landschaften entstehen: Wälder aus riesigen Lollipops und Seen aus glitzerndem Limonadenwasser, an denen kleine Roboter friedlich angelten. Jeder Meter der Fahrt fühlte sich an wie eine sanfte Schaukelbewegung, die ihre Augen lieder mit jedem Mal ein kleines Stückchen schwerer werden ließ.

Plötzlich hielt der Zug an einem Bahnhof, der ganz aus weichem Teddybär-Fell gebaut war und an dem kleine Püppchen mit Laternen warteten. Der Bahnhofsvorsteher, ein alter hölzerner Nussknacker, erklärte ihnen, dass die Schienen vor ihnen mit bunten Murmeln übersät seien, die sie erst wegräumen müssten. Leo und Max sprangen aus dem Waggon und begannen gemeinsam, die gläsernen Kugeln vorsichtig an den Rand zu rollen, wobei sie sich gegenseitig stützten und anfeuerten. Ihr Teamplay war so effizient, dass der Weg in kürzester Zeit wieder frei war und die Reise glücklich weitergehen konnte. Der Nussknacker schenkte ihnen zum Abschied zwei kleine Schlafmützen aus Seide, die sich wunderbar kühl auf ihren Stirnen anfühlten. Der Zug setzte seine Fahrt fort, und das rhythmische Klackern der Räder klang nun wie ein beruhigendes Metronom der Nacht.

Sie erreichten schließlich das Ende der Schienen, das direkt in eine riesige, flauschige Wolke führte, auf der ein Schloss aus Marshmallows thronte. Dort durften die Jungs aussteigen und in einem Meer aus Federn versinken, das so bequem war, dass sie sich nie wieder bewegen wollten. Das glückliche Ende ihrer Reise war erreicht, und die Dankbarkeit für diesen magischen Moment stand ihnen in die Gesichter geschrieben. Sie wussten, dass dieser Spielzeugzug immer für sie bereitstehen würde, solange sie ihre Fantasie bewahrten und als Brüder zusammenhielten. Die Welt der Träume hieß sie willkommen, und die Stille des Schlosses legte sich wie ein schützender Mantel um ihre kleinen Seelen. Es gab keinen Lärm mehr, nur noch das sanfte Leuchten der fernen Sterne, die durch die Fenster aus Zuckerglas schauten.

Die Rückfahrt verlief fast unbemerkt, da die Jungs bereits in ihren Sitzen zusammengesunken waren und leise vor sich hin schlummerten. Der magische Zug brachte sie sicher zurück in ihr Zimmer und schrumpfte pünchtlich zum ersten Sonnenstrahl wieder auf seine ursprüngliche Größe zusammen. Leo und Max lagen nun wieder in ihren eigenen Betten, eng aneinander gekuschelt, als hätten sie sich nie von der Stelle bewegt. Die Wärme ihrer Decken fühlte sich nun noch schöner an, und der Duft nach warmer Milch hing immer noch ganz leicht in der Zimmerluft. Es war ein Abenteuer, das sie in ihren Herzen bewahren würden, ein Beweis für ihre unzertrennliche Verbindung und ihren gemeinsamen Mut. Die Nacht war ihr Freund gewesen, und die Träume hatten sie mit offenen Armen empfangen.

Jetzt ruhen die zwei Helden Leo und Max in tiefem Frieden, während die Welt draußen langsam erwacht, ohne sie zu stören. Ihr Atem ist schwer und ruhig geworden, während die Schatten an der Wand friedliche Muster zeichnen, die zum Träumen einladen. Die Geschichte der Spielzeugbahn ist nun zu Ende, aber in ihren Herzen brennt das Licht des Abenteuers ewig weiter. Es ist die perfekte Zeit, um die Augen zu schließen und sich von der Sanftheit der Nacht davontragen zu lassen, genau wie die beiden Helden es getan haben. Die Stille im Haus ist vollkommen, und nur das Ticken der Uhr im Flur begleitet sie in den tiefen, erholsamen Schlaf. Spürst du auch schon, wie die Müdigkeit dich wie eine weiche Wolke umarmt und bist du bereit für dein eigenes Schloss?"""
    ).with_inputs("prompt_text"),
    dspy.Example(
        prompt_text="Tim und Jan erforschen den verzauberten Wald.",
        story="""### Der Hüter des verzauberten Waldes

Tim und sein älterer Bruder Jan standen am Rand eines Waldes, dessen Bäume so hoch waren, dass ihre Spitzen die Wolken kitzelten und silbern glänzten. Es hieß, dass in diesem Wald die schlafenden Riesen lebten, deren Schnarchen das sanfte Rauschen des Windes in den Blättern verursachte. Die beiden Jungs, die für ihren Mut und ihren Zusammenhalt im ganzen Dorf bekannt waren, wollten dem Hüter des Waldes ein Geschenk bringen. Jan trug eine Laterne, die mit gefangenen Glühwürmchen gefüllt war, während Tim einen Beutel mit magischen Kieselsteinen hielt, die im Dunkeln leuchteten. Gemeinsam setzten sie den ersten Fuß auf den moosigen Boden, der unter ihren Schritten nachgab wie eine weiche Matratze. Es war ein Ort voller Fantasie, an dem die Blumen flüsterten und die Pilze als kleine Laternen dienten, um ihnen den Weg zu weisen.

Ihr Weg führte sie tiefer in das Dickicht, wo die Farne wie große grüne Fächer wedelten und die Luft nach frischem Moos und Tannenharz duftete. Jan hielt die Laterne hoch, um den Pfad zu beleuchten, während Tim die Kieselsteine ausstreute, damit sie den Rückweg sicher wiederfinden konnten. Das Teamplay der zwei Brüder war instinktiv; sie wussten genau, wann der andere eine Pause brauchte oder eine helfende Hand beim Überqueren einer Wurzel. Sie begegneten einem alten Dachs, der ihnen den Weg zum Hüter wies und ihnen riet, ganz leise zu sein, um die Riesen nicht zu wecken. Die Stille des Waldes war nicht leer, sondern erfüllt von einem beruhigenden Summen, das wie eine sanfte Vibration in der Luft lag. Alles hier schien darauf ausgelegt zu sein, den Geist zur Ruhe zu bringen und die Sorgen des Tages zu vertreiben.

Schließlich erreichten sie eine Lichtung, in deren Mitte ein Baum stand, dessen Blätter aus reinem Silber bestanden und leise im Wind klangen. Dort saß der Hüter des Waldes, ein kleiner Geist mit einem Bart aus Flechten und gütigen, funkelnden Augen. Die Jungs überreichten ihr Geschenk, und der Hüter lächelte so breit, dass sein ganzes Gesicht in kleine Falten der Freude gelegt wurde. Als Dank berührte er ihre Stirnen mit einem silbernen Blatt, was ihnen ein Gefühl von absoluter Leichtigkeit und innerem Frieden schenkte. Das glückliche Ende ihrer Mission war erreicht, und sie spürten, wie die Müdigkeit nun wie ein schwerer, aber gemütlicher Mantel auf ihre Schultern sank. Der Hüter versprach ihnen, dass der Wald sie immer beschützen würde, solange sie so gute Freunde und Brüder blieben, wie sie es heute waren.

Der Rückweg fühlte sich viel kürzer an, da sie sich von der Magie des Waldes fast wie auf Schienen tragen ließen und ihre Schritte federleicht waren. Die leuchtenden Kieselsteine wiesen ihnen den Weg zurück zum Waldrand, wo der Mond bereits hoch am Himmel stand und das Dorf in ein friedliches Licht tauchte. Tim und Jan sprachen kaum noch, da sie die Ruhe des Waldes in sich aufgenommen hatten und ihre Gedanken bereits in Richtung ihrer weichen Betten wanderten. Die Welt um sie herum war still geworden, als hätten auch die Vögel und Rehe beschlossen, dass es Zeit für eine lange Pause sei. Das Abenteuer hatte ihre Herzen gestärkt und ihre Fantasie beflügelt, doch nun sehnten sie sich nach der Geborgenheit ihres Zuhauses. Es war ein schöner Abschluss für einen ereignisreichen Tag, der ihnen noch lange in Erinnerung bleiben würde.

Zuhause angekommen, schlüpften sie schnell aus ihren Schuhen und krochen unter die dicken Daunendecken, die sich heute besonders warm und einladend anfühlten. Jan löschte die Glühwürmchen-Laterne, und die kleinen Käfer flogen durch das offene Fenster zurück in die Freiheit des Gartens. Die beiden Brüder hielten sich noch kurz an den Händen, ein Zeichen ihres unerschütterlichen Zusammenhalts, bevor sie ihre Augen schlossen. Das Teamplay des Tages hatte sie müde, aber glücklich gemacht, und die Sicherheit ihres Zimmers war der beste Ort der Welt. Alles war nun in Ordnung, die Riesen schliefen friedlich weiter, und der Hüter des Waldes wachte über ihre Träume. Die Nacht hatte sich wie ein sanfter Kuss auf die Welt gelegt, und die Ruhe war vollkommen und ungestört.

Jetzt schlafen Tim und Jan tief und fest, während die silbernen Blätter des Hüters in ihren Träumen leise weiterklingen und ihnen Frieden schenken. Ihr Atem ist gleichmäßig und ruhig, ein Zeichen für einen tiefen und erholsamen Schlummer, den sie sich redlich verdient haben. Die Geschichte vom verzauberten Wald neigt sich dem Ende zu, aber die Magie bleibt in ihren Herzen lebendig, bis sie morgen wieder erwachen. Es gibt nichts mehr zu tun, außer loszulassen und sich von der Strömung der Nacht in das Land der Träume tragen zu lassen. Die Dunkelheit ist nicht erschreckend, sondern eine warme Wiege, die alle Sorgen des Tages sanft weggewischt hat. Kannst du das sanfte Schnarchen der Riesen ganz leise im Wind hören und bist du bereit für deine eigene Begegnung mit dem Hüter des Waldes?"""
    ).with_inputs("prompt_text"),
]

trainset = all_examples[:3]
valset = all_examples[3:]

# --- 7. OPTIMIERUNG ---
print("Starte GEPA Optimierung mit Best-Prompt-Tracking...")

with mlflow.start_run():
    mlflow.log_param("optimizer", "GEPA")
    mlflow.log_param("max_tokens", 4000)
    
    optimizer = GEPA(
        metric=story_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=25,
        num_threads=1 # Nur 1 Thread um Ressourcen/Timeouts zu schonen
    )

    try:
        optimized_student = optimizer.compile(StoryStudent(), trainset=trainset, valset=valset)
        
        # Finalen Stand sichern
        final_instr = optimized_student.predictor.signature.instructions
        mlflow.log_text(final_instr, "final_optimized_prompt.txt")
        with open("best_prompt.txt", "a", encoding="utf-8") as f:
            f.write("\n\n" + "="*40 + "\nFINALER OPTIMIERTER PROMPT:\n" + final_instr)
            
        print("\nPROMPT OPTIMIERT. Höchster Score in best_prompt.txt und MLflow.")
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    res = optimized_student(prompt_text="Zwei Brüder erforschen eine alte Burgruine.")
    print(f"\nFINALE STORY:\n{res.story}")