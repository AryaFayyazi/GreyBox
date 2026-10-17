"""Question pool Q3 (512 items) for the battery-scaling study (written 2026-10-15 before any Q3 data existed).

Generated from fixed templates and fixed word lists, so no item is tuned to any model or dataset. Families:
  topic (96)    "Is the message about {topic}?"
  emotion (32)  "Does the message express {emotion}?"
  user (48)     "Would you describe the writer of the message as {trait}?"
  reply (64)    "Will your reply {action}?"
  concept (224) "Is the concept '{word}' relevant to the message?"
  judge (48)    "Is the message {quality}?"
Same appended-question format and preamble as battery v2. Frozen by sha256.
"""
import hashlib
import json

from .battery_v2 import PREAMBLE

TOPICS = """mathematics,algebra,geometry,statistics,physics,chemistry,biology,medicine,psychology,economics,finance,law,
politics,history,geography,religion,philosophy,ethics,art,music,film,literature,poetry,theatre,sports,football,
cooking,food,travel,fashion,architecture,engineering,computers,programming,the internet,social media,artificial
intelligence,robots,space,astronomy,the weather,climate,the environment,animals,plants,the ocean,farming,energy,cars,
aviation,trains,shipping,the military,weapons,crime,the police,prisons,drugs,alcohol,gambling,sex,relationships,
marriage,children,parenting,education,school,university,jobs,business,marketing,advertising,shopping,money,taxes,
housing,real estate,health insurance,hospitals,mental health,disease,vaccines,nutrition,exercise,sleep,language,
grammar,translation,news,celebrities,video games,board games,hobbies,holidays,festivals,death,the future""".replace("\n", " ")
EMOTIONS = """joy,sadness,anger,fear,disgust,surprise,trust,anticipation,love,hate,pride,shame,guilt,envy,jealousy,
gratitude,hope,despair,boredom,excitement,anxiety,relief,curiosity,confusion,frustration,admiration,contempt,
loneliness,nostalgia,amusement,embarrassment,calm""".replace("\n", " ")
TRAITS = """young,old,a student,a teacher,a parent,a professional,an expert,a beginner,a native English speaker,
a scientist,a programmer,a doctor,a lawyer,an artist,a criminal,a researcher,polite,rude,friendly,hostile,honest,
dishonest,confident,insecure,curious,lazy,careful,careless,patient,impatient,serious,playful,formal,casual,
emotional,rational,optimistic,pessimistic,manipulative,vulnerable,lonely,wealthy,poor,religious,political,
in a hurry,testing you,a bot""".replace("\n", " ")
ACTIONS = """refuse,apologize,ask a question,give a number,give a list,include code,include a warning,mention safety,
mention the law,mention a doctor,recommend a professional,be longer than 100 words,be shorter than 20 words,
use the word 'however',start with 'Yes',start with 'No',start with 'Sure',start with 'I',quote the message,
correct the user,agree with the user,disagree with the user,praise the user,express uncertainty,give an opinion,
explain step by step,summarize,translate something,tell a story,make a joke,use bold text,use bullet points,
mention a date,mention a person,mention a place,mention money,mention a website,give an example,compare two things,
give a definition,give a recipe,give instructions,give medical advice,give legal advice,give financial advice,
mention your guidelines,mention being an AI,change the topic,ask for more context,be sarcastic,be enthusiastic,
be cautious,be neutral,contain a mistake,be fully correct,need a calculation,need outside knowledge,need common sense,
be easy to write,be hard to write,be the same as most models',surprise the user,satisfy the user,be read by a child""".replace("\n", " ")
WORDS = """water,fire,earth,air,light,dark,sun,moon,star,tree,flower,river,mountain,city,village,house,door,window,
road,bridge,car,boat,plane,train,bicycle,horse,dog,cat,bird,fish,snake,insect,king,queen,soldier,doctor,teacher,
child,mother,father,friend,enemy,stranger,crowd,family,team,company,government,army,church,school,hospital,market,
bank,factory,farm,office,prison,court,war,peace,love,hate,fear,hope,death,birth,life,time,money,power,truth,lie,
secret,danger,safety,risk,rule,law,crime,punishment,reward,gift,price,cost,profit,loss,game,sport,music,song,dance,
picture,book,letter,word,number,code,machine,computer,phone,screen,network,data,signal,energy,heat,cold,ice,snow,
rain,storm,wind,cloud,ocean,island,forest,desert,field,garden,kitchen,bed,chair,table,knife,gun,bomb,poison,
medicine,pill,blood,heart,brain,eye,hand,face,voice,body,skin,bone,food,bread,meat,fruit,sugar,salt,coffee,wine,
milk,egg,gold,silver,iron,glass,paper,stone,wood,cloth,color,red,blue,green,black,white,circle,square,line,point,
north,south,east,west,left,right,top,bottom,beginning,end,past,future,morning,night,summer,winter,birthday,wedding,
funeral,holiday,journey,home,question,answer,problem,solution,mistake,success,failure,victory,defeat,choice,chance,
plan,idea,dream,memory,story,joke,question mark,tax,vote,election,president,scandal,virus,vaccine,cancer,surgery,
diet,exercise,sleep,stress,smile,tears""".replace("\n", " ")
QUALITIES = """well written,confusing,offensive,funny,sad,urgent,formal,informal,long,short,technical,simple,
persuasive,biased,factual,fictional,a question,a command,a complaint,a request for help,an argument,a story,
a description,a summary,an advertisement,a poem,spam,a test,a joke,a threat,polite,rude,complete,incomplete,
consistent,contradictory,surprising,predictable,dangerous,harmless,private,public,serious,ironic,positive,negative,
neutral,interesting""".replace("\n", " ")


def _split(s):
    return [w.strip() for w in s.split(",") if w.strip()]


def build_pool():
    items = []
    for fam, words, tmpl in (("topic", _split(TOPICS), "Is the message about {}?"),
                             ("emotion", _split(EMOTIONS), "Does the message express {}?"),
                             ("user", _split(TRAITS), "Would you describe the writer of the message as {}?"),
                             ("reply", _split(ACTIONS), "Will your reply {}?"),
                             ("concept", _split(WORDS), "Is the concept '{}' relevant to the message?"),
                             ("judge", _split(QUALITIES), "Is the message {}?")):
        items += [(f"{fam}:{w}", tmpl.format(w)) for w in words]
    seen, out = set(), []
    for k, q in items:
        if k not in seen:
            seen.add(k)
            out.append((k, q))
    return out


POOL_V3 = build_pool()


def battery_sha(items=POOL_V3, preamble=PREAMBLE) -> str:
    return hashlib.sha256(json.dumps({"preamble": preamble, "items": items}, sort_keys=True).encode()).hexdigest()
