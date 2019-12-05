import basilica

sentences = [
    "This is a sentence!",
    "This is a similar sentence!",
    "I don't think this sentence is very similar at all...",
]

def test(auth_key):
    with basilica.Connection(auth_key) as c:
        embeddings = list(c.embed_sentences(sentences))    
    return embeddings

if __name__ == '__main__':
    auth_key = sys.argv[1]
    try:
        test(auth_key)
        assert len(embeddings) == len(sentences)
    except Excetpion as e:
        return e
    else:
        return True