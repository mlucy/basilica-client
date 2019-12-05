import basilica
import time
import unittest
from scipy import spatial
from six.moves.queue import Queue
import os

test_key = os.environ.get('BASILICA_TEST_KEY', 'SLOW_DEMO_KEY')
fake_key = 'FAKE_KEY'
sentences_small = [
    "This is a sentence!",
    "This is a similar sentence!",
    "I don't think this sentence is very similar at all...",
]

sentences_large = [
    "This is a sentence!",
    "This is a similar sentence!",
    "I don't think this sentence is very similar at all...",
] * 256

def test_testkey_small():
    print("Test embedding 3 sentences with TEST_KEY")
    begin = time.time()
    try:
        with basilica.Connection(test_key) as c:
            embeddings = list(c.embed_sentences(sentences_small))
    except Exception as err:
        print("Exception rasied : " + str(err))
    else:
        assert len(embeddings)==3
        assert all([len(e) == 512 for e in embeddings])
        print("No exception raised, test passed")
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_fakekey_small():
    print("\nTest embedding 3 sentences with fake_key")
    print("Expecting HTTPError")
    begin = time.time()
    try:
        with basilica.Connection(fake_key) as c:
            embeddings = list(c.embed_sentences(sentences_small))
    except Exception as err:
        print("Error raised : " + str(err))
    else:
        print("Test failed, embeddings returend with length of ", len(embeddings))
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_demokey_small():
    print("\nTest embedding 3 sentences with SLOW_DEMO_KEY")
    begin = time.time()
    try:
        with basilica.Connection('SLOW_DEMO_KEY') as c:
            embeddings = list(c.embed_sentences(sentences_small))
    except Exception as err:
        print(err)
    else:
        assert len(embeddings)==3
        assert all([len(e) == 512 for e in embeddings])
        print("No exception raised, test passed")
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_testkey_large():
    print("\nTest embedding 768 sentences with TEST_KEY")
    begin = time.time()
    try:
        with basilica.Connection(test_key) as c:
            embeddings = list(c.embed_sentences(sentences_large))
    except Exception as err:
        print("Test failed, Exception raised : " + str(err))
    else:
        assert len(embeddings)==768
        assert all([len(e) == 512 for e in embeddings])
        print("No exception raised, test passed")
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_fakekey_large():
    print("\nTest embedding 768 sentences with fake_key")
    print("Expecting HTTPError")
    begin = time.time()
    try:
        with basilica.Connection(fake_key) as c:
            embeddings = list(c.embed_sentences(sentences_large))
    except Exception as err:
        print("Exception rasied : " + str(err))
    else:
        print("Test failed, embeddings returend with length of ", len(embeddings))
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_timeout(timeout=0.01):
    print("\nTest timeout")
    print("Expecting Exeptions")
    begin = time.time()
    try:
        with basilica.Connection(test_key) as c:
            embeddings = list(c.embed_sentences(sentences_large, timeout=timeout))
    except Exception as err:
        print("Exception raised:" + str(err))
    else:
        print("No exception raised, test failed")
    t1 = time.time() - begin
    print("time took : " + str(t1))

def test_exception():
    print("\nTest expecting Exceptions")
    with basilica.Connection(test_key) as c:
        embeddings = list(c.embed_sentences(sentences_small))
    print("No exception raised, embeddings returned "+str(len(embeddings)))

def gen(s):
    for i in s:
        yield i

def sufficiently_equal(v1, v2):
    return spatial.distance.cosine(v1, v2) < 1e-9

def test_sameconnection():
    print("\nTest concurrent embedding calls")
    sentences_outer = ["This is a sentence!"] * 39
    sentences_inner = ["This sentence not so same?"] * 39
    sentences_truth = ["This is a sentence!", "This sentence not so same?"]
    # getting ground-truth
    with basilica.Connection(test_key) as c:
        sentences_truth = list(c.embed_sentences(sentences_truth))
    # Created another connection
    emb_outer = []
    emb_inner = []
    with basilica.Connection(test_key) as c:
        for x in c.embed_sentences(gen(sentences_outer)):
            emb_outer.append(x)
            for y in c.embed_sentences(gen(sentences_inner)):
                emb_inner.append(y)
    assert all([sufficiently_equal(sentences_truth[0], e) for e in emb_outer])
    assert all([sufficiently_equal(sentences_truth[1], e) for e in emb_inner])
    print("Test Passed!")

if __name__ == "__main__":
    test_testkey_small()
    test_fakekey_small()
    test_demokey_small()
    test_testkey_large()
    test_fakekey_large()
    test_timeout()
    # test_exception()
    test_sameconnection()
