import basilica
import unittest
from requests import HTTPError

test_key = 'ec6a2eec-94c7-4f9b-e656-c0d0824b60fe'
fake_key = 'FAKE_KEY'
sentences = [
    "This is a sentence!",
    "This is a similar sentence!",
    "I don't think this sentence is very similar at all...",
]

class basilicaTesting(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.connection = basilica.Connection(test_key)

    def test_Connection(self):
        self.assertIsInstance(self.connection, basilica.Connection)
    
    def test_typeChecking(self):
        embedding = self.connection.embed_sentences(sentences)
        self.assertIsInstance(embedding, types.GeneratorType)
        self.assertEqual(len(list(embedding)), len(sentences))

    def test_HTTPError(self):
        with basilica.Connection(fake_key) as c:
            try:
                c.embed_sentences(sentences).next()
            except Exception as e:
                self.assertIsInstance(e, HTTPError)

    # same connection
    def

if __name__ == '__main__':
    unittest.main()