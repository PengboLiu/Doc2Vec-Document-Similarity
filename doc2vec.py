import gensim
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_corpus():

    with open("corpus_seg.txt", 'r') as doc:
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_doc2vec')
    return model_dm

def test():
    model_dm = Doc2Vec.load("model_doc2vec")
    text_test = u'武汉东湖新技术开发区人民检察院指控： 2013年4月27日21时许，被告人王某、连某经预谋后，窜至本区流芳高新四路联想工地内，窃取该处扣件若干欲离开时，被此处工地值班人员刘某发现并制止。被告人王某、连某遂共同用拳头、安全帽及啤酒瓶殴打刘某的头部、背部等处，致被害人刘某轻微伤，后共同逃离现场。 2013年11月1日，被告人王某被公安机关抓获。同年11月25日，被告人王某按照公安机关的安排，以打电话的方式联系被告人连某投案。到案后，上述二被告人共同赔偿被害人刘某人民币1．5万元，并获得谅解。 针对上述指控的事实，公诉机关当庭出示和宣读的证据有：1、抓获及破案经过；2、调解协议、谅解书、病历等书证；3、涉案物品照片；4、鉴定意见书；5、证人证言；6、被害人陈述；7、被告人的供述及辩解、讯问同步录音录像等。 公诉机关认为，被告人王某、连某以非法占有为目的，在实施盗窃行为时，为抗拒抓捕，当场使用暴力，致一人轻微伤，其行为均触犯了《中华人民共和国刑法》第二百六十九条、第二百六十三条的规定，应当以抢劫罪追究其刑事责任。案发后，被告人王某协助公安机关抓捕同案犯，具有《中华人民共和国刑法》第六十八条规定的情节；被告人连某主动投案，并如实供述自己的犯罪事实，具有《中华人民共和国刑法》第六十七条第一款规定的情节。'
    text_cut = jieba.cut(text_test)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims


if __name__ == '__main__':
    x_train = get_corpus()
    # model_dm = train(x_train)
    sims = test()
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, sim, len(sentence[0]))