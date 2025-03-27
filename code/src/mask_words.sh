#!/bin/bash


bbase_words=(
    "white" "beaners" "goatfuckers" "oreo"
    "like" "vitriol" "mumbles"
    "nigger" "kike" "moots"
    "people" "yadi" "women" "topics"
    "multifandom" "aromantic" "kafir"
    "jews" "standwith" "hoes"
    "ghetto" "replying" "niggerish" "lebron"
    "fucking" "niglets" "nigress" "avocados"
    "retarded" "bluetooth" "one" "kashmirs"
    "carmelo" "black" "motherfucker"
    "tiki" "muslim" "muzzie"
    "fuck" "beaner" "foolish" "hate"
    "negresses" "bike" "edgelords" "raghead"
    "rey"
    "accent"
    "polish"
    "dial"
    "exams"
    "gies"
    "dresses"
    "partridge"
    "culturally"
    "sherry"
    "hbo"
    "valerie"
    "nude"
    "cathy"
)


berta_words=(
    "white" "beaners" "goatfuckers" "partys"
    "like" "mudslimes" "vitriol" "noon"
    "nigger" "muslimes" "people"
    "niglets" "retarded" "women" "faggotry"
    "retard" "kike" "agains" "jews"
    "retards" "ghetto" "beaner" "nails"
    "fucking" "chink" "uplifting" "brighton"
    "niglet" "one"
    "negresses" "tradition" "transgenderism"
    "black" "spic" "barely"
    "muslim" "niggers" "anglin" "coherent"
    "fuck" "muzzies" "faggotry"
    "hate" "camel" "tommy" "defame"
    "listing"
    "dorothy"
    "culturally"
    "malaysia"
    "experiences"
    "stage"
    "subtweeting"
    "lighting"
    "victimized"
    "eddie"
    "insufferable"
    "hex"
    "cookie"
    "complained"
)

for word in "${bbase_words[@]}"; do
    python3 masking_dataset_byword.py "$word" bbase
done


for word in "${berta_words[@]}"; do
    python3 masking_dataset_byword.py "$word" distilroberta
done
