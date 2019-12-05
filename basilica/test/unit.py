from scipy import spatial
from six.moves.queue import Queue
import basilica
import os
import requests
import six
import time
import unittest
import unittest

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

class TestDriver(unittest.TestCase):
    def test_testkey_small(self):
        with basilica.Connection(test_key) as c:
            embeddings = list(c.embed_sentences(sentences_small))
        self.assertEqual(3, len(embeddings))
        for e in embeddings:
            self.assertEqual(512, len(e))

    def test_fakekey_small(self):
        with six.assertRaisesRegex(self, requests.exceptions.HTTPError, r"^401 Client Error: Unauthorized.*$"):
            with basilica.Connection(fake_key) as c:
                embeddings = list(c.embed_sentences(sentences_small))

    def test_testkey_large(self):
        with basilica.Connection(test_key) as c:
            embeddings = list(c.embed_sentences(sentences_large))
        self.assertEqual(768, len(embeddings))
        for e in embeddings:
            self.assertEqual(512, len(e))

    def test_fakekey_large(self):
        with six.assertRaisesRegex(self, requests.exceptions.HTTPError, r"^401 Client Error: Unauthorized.*$"):
            with basilica.Connection(fake_key) as c:
                embeddings = list(c.embed_sentences(sentences_large))

    def test_timeout(self):
        with six.assertRaisesRegex(self, requests.exceptions.ReadTimeout, r"^.*Read timed out.*$"):
            with basilica.Connection(test_key) as c:
                embeddings = list(c.embed_sentences(sentences_large, timeout=0.1))

    def test_sameconnection(self):
        def gen(s):
            for i in s:
                yield i

        def sufficiently_equal(v1, v2):
            return spatial.distance.cosine(v1, v2) < 1e-9

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
        for e in emb_outer:
            self.assertTrue(sufficiently_equal(sentences_truth[0], e))
        for e in emb_inner:
            self.assertTrue(sufficiently_equal(sentences_truth[1], e))

if __name__ == "__main__":
    unittest.main()
