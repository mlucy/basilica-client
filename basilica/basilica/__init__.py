import requests
import base64

class Connection(object):
    def __init__(self, auth_key, server='https://api.basilica.ai'):
        """A connection to basilica.ai that can be used to generate embeddings.

        :param auth_key: Your auth key.  You can view your auth keys at https://basilica.ai/auth_keys.
        :type auth_key: str
        :param server: What URL to use to connect to the server.
        :type server: str

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_sentence('A sentence.'))
        [0.6246702671051025, ..., -0.03025037609040737]
        """
        self.server = server
        self.session = requests.Session()
        self.session.auth = (auth_key, '')

    def __enter__(self, *a, **kw):
        self.session.__enter__(*a, **kw)
        return self

    def __exit__(self, *a, **kw):
        return self.session.__exit__(*a, **kw)

    def raw_embed(self, url, data, opts):
        if type(url) != str:
            raise ValueError('`url` argument must be a string (got `%s`)' % url)
        if type(opts) != dict:
            raise ValueError('`url` argument must be a dict (got `%s`)' % url)
        if 'data' in opts:
            raise ValueError('`opts` argument may not contain `data` key (got `%s`)' % opts)
        query = opts.copy()
        query['data'] = data
        res = self.session.post(url, json=query)
        res.raise_for_status()
        out = res.json()
        if 'error' in out:
            raise RuntimeError('basilica.ai server returned error: `%s`' % out['error'])
        if 'embeddings' not in out:
            raise RuntimeError('basilica.ai server did not return embeddings: `%s`' % out)
        return out['embeddings']

    # TODO: parallelize
    def embed(self, url, data, batch_size, opts):
        batch = []
        for i in data:
            batch.append(i)
            if len(batch) >= batch_size:
                for e in self.raw_embed(url, batch, opts=opts):
                    yield e
                batch = []
        if len(batch) > 0:
            for e in self.raw_embed(url, batch, opts=opts):
                yield e
            batch = []

    def embed_images(self, images, model='generic', version='default', batch_size=64, opts={}):
        """Generate embeddings for images.  Images should be passed as byte strings, and will be sent to the server in batches to be embedded.

        :param images: An iterable (such as a list) of the images to embed.
        :type images: Iterable[str]
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server in a batch.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: A generator of embeddings.
        :rtype: Generator[List[float]]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   images = []
        ...   for filename in ['img1.jpg', 'img2.jpg']:
        ...     with open(filename, 'rb') as f:
        ...     images.append(f.read())
        ...   for embedding in c.embed_images(images):
        ...     print(embedding)
        [0.6246702671051025, ...]
        [-0.03025037609040737, ...]
        """
        url = '%s/embed/images/%s/%s' % (self.server, model, version)
        data = ({'img': base64.b64encode(img).decode('utf-8')} for img in images)
        return self.embed(url, data, batch_size=batch_size, opts=opts)

    def embed_image(self, image, model='generic', version='default', opts={}):
        """Generate the embedding for an image.  The image should be passed as a byte string.

        :param image: The image to embed.
        :type image: str
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   with open('img.jpg', 'rb') as f:
        ...     print(c.embed_image(f.read()))
        [0.6246702671051025, ...]
        """
        return list(self.embed_images([image], model=model, version=version, opts=opts))[0]

    def embed_image_files(self, image_files, model='generic',
                          version='default', batch_size=64, opts={}):
        """Generate embeddings for image files.  The file names should be passed as paths that can be understood by `open`.

        :param image_files: An iterable (such as a list) of paths to the images to embed.
        :type image_files: Iterable[str]
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server in a batch.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: A generator of embeddings.
        :rtype: Generator[List[float]]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   for embedding in c.embed_image_files(['img1.jpg', 'img2.jpg']):
        ...     print(embedding)
        [0.6246702671051025, ...]
        [-0.03025037609040737, ...]
        """
        def load_image_files(image_files):
            for image_file in image_files:
                with open(image_file, 'rb') as f:
                    yield f.read()
        return self.embed_images(load_image_files(image_files),
                                 model=model, version=version, batch_size=batch_size, opts=opts)

    def embed_image_file(self, image_file, model='generic', version='default', opts={}):
        """Generate the embedding for an image file.  The file name should be passed as a path that can be understood by `open`.

        :param image_file: Path to the image to embed.
        :type image_file: str
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_image_file('img.jpg')
        [0.6246702671051025, ...]
        """
        with open(image_file, 'rb') as f:
            return self.embed_image(f.read(), model=model, version=version, opts=opts)

    def embed_sentences(self, sentences, model='english', version='default',
                        batch_size=64, opts={}):
        """Generate embeddings for sentences.

        :param sentences: An iterable (such as a list) of sentences to embed.
        :type sentences: Iterable[str]
        :param model: What model to use (i.e. the kind of sentence being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server in a batch.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: A generator of embeddings.
        :rtype: Generator[List[float]]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   for embedding in c.embed_sentences(['Sentence one.', 'Sentence two.']):
        ...     print(embedding)
        [0.6246702671051025, ...]
        [-0.03025037609040737, ...]
        """
        url = '%s/embed/text/%s/%s' % (self.server, model, version)
        data = sentences
        return self.embed(url, data, batch_size=batch_size, opts=opts)

    def embed_sentence(self, sentence, model='english', version='default', opts={}):
        """Generate the embedding for a sentence.

        :param sentence: The sentence to embed.
        :type sentence: str
        :param model: What model to use (i.e. the kind of sentence being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_sentence('This is a sentence.')
        [0.6246702671051025, ...]
        """
        return list(self.embed_sentences([sentence], model=model, version=version,
                                         batch_size=batch_size, opts=opts))[0]
