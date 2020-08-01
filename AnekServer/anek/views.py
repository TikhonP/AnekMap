from rest_framework import viewsets
# from rest_framework import permissions
from anek.serializers import AnekSerializer, TagSerializer
from anek.models import Anek, Tag
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import dateutil.parser
import json
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK
)
from rest_framework.response import Response


from ufal.udpipe import Model, Pipeline
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import models
import numpy as np
import scipy.spatial.distance as ds
from sklearn.neighbors import NearestNeighbors

import zipfile
from gensim import models

model_file = '182.zip'

with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = models.KeyedVectors.load_word2vec_format(stream, binary=True)

with open('/Users/tikhon/Downloads/totikhon2.json') as f:
    all_aneks = json.load(f)

def clean_token(token, misc):
    """
    :param token:  токен (строка)
    :param misc:  содержимое поля "MISC" в CONLLU (строка)
    :return: очищенный токен (строка)
    """
    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def clean_lemma(lemma, pos):
    """
    :param lemma: лемма (строка)
    :param pos: часть речи (строка)
    :return: очищенная лемма (строка)
    """
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])
        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])
        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])
    return out_lemma
def process(pipeline, text='Строка', keep_pos=True, keep_punct=False):
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []

    # обрабатываем текст, получаем результат в формате conllu:
    processed = pipeline.process(text)

    # пропускаем строки со служебной информацией:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        token = clean_token(token, misc)
        lemma = clean_lemma(lemma, pos)
        if not lemma or not token:
            continue
        if pos in entities:
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))
        else:
            if not named:
                if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                    lemma = num_replace(token)
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))

    if not keep_punct:
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos:
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    return tagged_propn


def count2doc(doc, model):
    documents = doc.split("\n")
    mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]
    tfidf = models.TfidfModel(corpus, smartirs='ntc')

    tfidf_dict = {}

    for doc in tfidf[corpus]:
        for idd, freq in doc:
            tfidf_dict[proctext(mydict[idd])[0]]=freq
    w2v_dict = {}

    for word in tfidf_dict:
        try:
            vec = model[word]
        except KeyError:
            pass
            continue
        newvec = vec*tfidf_dict[word]
        w2v_dict[word] = newvec

    w2v_list = np.zeros(300,)

    for word in w2v_dict:
        w2v_list = w2v_list + w2v_dict[word]


    return w2v_list
# udpipe_model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
# udpipe_filename = udpipe_model_url.split('/')[-1]

modelprep = Model.load('udpipe_syntagrus.model')
process_pipeline = Pipeline(modelprep, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

def proctext(text):
    return process(process_pipeline, text)

neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nbrs = neigh.fit(np.array([x["vec2"] for x in all_aneks]))
def predict_coords(query, model):
    queryvec = count2doc(str(query), model)
    queryvec = np.concatenate((queryvec, np.zeros((157,), dtype=np.float32)), axis=0)
    ind = nbrs.kneighbors([queryvec])[1][0][0]
    return all_aneks[ind]['2dim_vec2']




class AnekViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Anek to be viewed or edited.
    """
    queryset = Anek.objects.all()
    serializer_class = AnekSerializer
    # permission_classes = [permissions.IsAuthenticated]


class TagViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Tag to be viewed or edited.
    """
    queryset = Tag.objects.all()
    serializer_class = TagSerializer

"""
@csrf_exempt
def getAneksForCanvas(request):

    API endpoint that allows Aneks to be get by range X, Y and time

    REQUEST json:
        {"from_x": Int,
         "to_x": Int,
         "from_y": Int,
         "to_y": Int,
         "from_date": DateTime,
         "to_date": DateTime,
        }
    ANSWER is list of dicts with aneks
    if request.method == 'GET':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'status': 0, 'error': 'Invalid body json or empty body'})
        from_d = dateutil.parser.parse(data.get('from_date', None))
        to_d = dateutil.parser.parse(data.get('to_date', None))

        aneks_out = Anek.objects.filter(
            date_created__gte=from_d,
            date_created__lte=to_d,
            x__gte=data.get('from_x', None),
            x__lte=data.get('to_x', None),
            y__gte=data.get('from_y', None),
            y__lte=data.get('to_y', None),
        )

        out_json = []

        for i in aneks_out:
            out_json.append({
                "text_preview": i.text_preview,
                "text": i.text,
                "date_created": i.date_created,
                "x": i.x,
                "y": i.y,
                "views": i.views,
                "href": i.href,
            })

        return Response(out_json, status=HTTP_200_OK)
    else:
        return JsonResponse({'status': 0, 'error': 'Invalid request method ({}). Must be GET.'.format(request.method)})
"""

@csrf_exempt
def search(request):
    """
    API endpoint that allows to search aneks by string

    REQUEST get param:
        query: string
    ANSWER json:
        {"x" Int,
         "y": Int}
    """

    if request.method == 'GET':
        query = request.GET.get('query', None)
        if query is None:
            return JsonResponse({'status': 0, 'error': 'Must contain query param'})
        a = predict_coords(query, model)
        return JsonResponse({'x': a[0], 'y': a[1]})

    else:
        return JsonResponse({'status': 0, 'error': 'Invalid request method ({}). Must be GET.'.format(request.method)})

@csrf_exempt
def getAllAneksWithLabels(request):
    if request.method == 'GET':
        queryset = Anek.objects.all()

        out_json = []

        for i in queryset:
            out_json.append({
                "text_preview": i.text_preview,
                "text": i.text,
                # "date_created": i.date_created,
                "x": i.x,
                "y": i.y,
                "views": i.views,
                "href": i.href,
                "tags": list(i.tag_set.all())
            })

        return JsonResponse(out_json, )

    else:
        return JsonResponse({'status': 0, 'error': 'Invalid request method ({}). Must be GET.'.format(request.method)})

