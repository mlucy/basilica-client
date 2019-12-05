from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import base64
import requests
import io
from PIL import Image
import threading
from six.moves.queue import Queue, Empty

__version__ = '0.2.7'

class Connection(object):
    def __init__(self, auth_key, server='https://api.basilica.ai',
                 retries=2, backoff_factor=0.1, status_forcelist=(500)):
        """A connection to basilica.ai that can be used to generate embeddings.

        :param auth_key: Your auth key.  You can view your auth keys at https://basilica.ai/api-keys/.
        :type auth_key: str
        :param server: What URL to use to connect to the server.
        :type server: str
        :param retries: Number of times to retry failed connections and requests.
        :type retries: int
        :param backoff_factor: See urllib3.util.retry.Retry.backoff_factor .
        :type backoff_factor: float
        :param status_forcelist: What HTTP response codes trigger a retry.
        :type status_forcelist: Tuple[int]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_sentence('A sentence.'))
        [0.6246702671051025, ..., -0.03025037609040737]
        """
        self.server = server
        self.session = requests.Session()
        self.session.auth = (auth_key, '')

        self.retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        self.adapter = HTTPAdapter(max_retries=self.retry)
        self.session.mount('http://', self.adapter)
        self.session.mount('https://', self.adapter)

    def __enter__(self, *a, **kw):
        self.session.__enter__(*a, **kw)
        return self

    def __exit__(self, *a, **kw):
        return self.session.__exit__(*a, **kw)

    def raw_embed(self, url, data, opts, timeout):
        if type(url) != str:
            raise ValueError('`url` argument must be a string (got `%s`)' % url)
        if type(opts) != dict:
            raise ValueError('`url` argument must be a dict (got `%s`)' % url)
        if 'data' in opts:
            raise ValueError('`opts` argument may not contain `data` key (got `%s`)' % opts)
        query = opts.copy()
        query['data'] = data
        # For some reason the requests library doesn't retry timeouts
        # on its own.  We don't bother with backoff.
        for i in range(self.retry.read+1):
            try:
                headers = { 'User-Agent': 'Basilica Python Client (%s)' % __version__ }
                res = self.session.post(url, json=query, timeout=timeout, headers=headers)
            except requests.exceptions.Timeout:
                if i < self.retry.read:
                    continue
                else:
                    raise
            break
        res.raise_for_status()
        out = res.json()
        if 'error' in out:
            raise RuntimeError('basilica.ai server returned error: `%s`' % out['error'])
        if 'embeddings' not in out:
            raise RuntimeError('basilica.ai server did not return embeddings: `%s`' % out)
        return out['embeddings']

    def embed(self, url, data, batch_size, opts, timeout):
        batch_queue = Queue(maxsize=1)
        emb_queue = Queue()
        api_thread = threading.Thread(target=self.raw_embed_wrapper, args=(url, opts, timeout, batch_queue, emb_queue))
        api_thread.setDaemon(True)
        api_thread.start()
        batch = []
        for i in data:
            batch.append(i)
            if len(batch) >= batch_size:
                try:
                    emb = emb_queue.get(block=False)
                    if isinstance(emb, Exception):
                        batch_queue.put('DONE', block=True)
                        raise emb
                    else:
                        for e in emb:
                            yield e
                except Empty:
                    pass
                batch_queue.put(batch, block=True)
                batch = []
        if len(batch) > 0:
            batch_queue.put(batch, block=True)
        batch_queue.put('DONE', block=True)
        while True:
            emb = emb_queue.get(block=True)
            if isinstance(emb, Exception):
                raise emb
            elif emb == 'DONE':
                break
            else:
                for e in emb:
                    yield e

    def raw_embed_wrapper(self, url, opts, timeout, batch_queue, emb_queue):
        while True:
            try:
                batch = batch_queue.get(block=True)
                if batch == 'DONE':
                    emb_queue.put('DONE')
                    return None
                emb = self.raw_embed(url, batch, opts=opts, timeout=timeout)
                emb_queue.put(emb)
            except Exception as err:
                emb_queue.put(err)

    def embed_images(self, images, model='generic', version='default',
                     batch_size=32, opts={}, timeout=30):
        """Generate embeddings for JPEG images.  Images should be passed as byte strings, and will be sent to the server in batches to be embedded.

        :param images: An iterable (such as a list) of the images to embed.
        :type images: Iterable[str]
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server at a time.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool
        :param timeout: HTTP timeout for request.
        :type timeout: int
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
        data = ({'img': self.__encode_image(img, transform_image=opts.get("transform_image", True) )} for img in images)
        return self.embed(url, data, batch_size=batch_size, opts=opts, timeout=timeout)

    def embed_image(self, image, model='generic', version='default',
                    opts={}, timeout=10):
        """Generate the embedding for a JPEG image.  The image should be passed as a byte string.

        :param image: The image to embed.
        :type image: str
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool
        :param timeout: HTTP timeout for request.
        :type timeout: int
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   with open('img.jpg', 'rb') as f:
        ...     print(c.embed_image(f.read()))
        [0.6246702671051025, ...]
        """
        return list(self.embed_images([image], model=model, version=version,
                                      opts=opts, timeout=timeout))[0]

    def embed_image_files(self, image_files, model='generic', version='default',
                          batch_size=32, opts={}, timeout=30):
        """Generate embeddings for JPEG image files.  The file names should be passed as paths that can be understood by `open`.

        :param image_files: An iterable (such as a list) of paths to the images to embed.
        :type image_files: Iterable[str]
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server at a time.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool
        :param timeout: HTTP timeout for request.
        :type timeout: int
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
        return self.embed_images(load_image_files(image_files), model=model, version=version,
                                 batch_size=batch_size, opts=opts, timeout=timeout)

    def embed_image_file(self, image_file, model='generic', version='default',
                         opts={}, timeout=10):
        """Generate the embedding for a JPEG image file.  The file name should be passed as a path that can be understood by `open`.

        :param image_file: Path to the image to embed.
        :type image_file: str
        :param model: What model to use (i.e. the kind of image being embedded).
        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool

        :param timeout: HTTP timeout for request.
        :type timeout: int
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_image_file('img.jpg')
        [0.6246702671051025, ...]
        """
        with open(image_file, 'rb') as f:
            return self.embed_image(f.read(), model=model, version=version,
                                    opts=opts, timeout=timeout)

    def embed_sentences(self, sentences, model='english', version='default',
                        batch_size=64, opts={}, timeout=15):
        """Generate embeddings for sentences.

        :param sentences: An iterable (such as a list) of sentences to embed.
        :type sentences: Iterable[str]
        :param model: What model to use (i.e. the kind of sentence being embedded).

            * **generic:** Generic English text embedding (the default.)
            * **reddit:** Text embedding specialized for English Reddit posts.
            * **twitter:** Text embedding specialized for English tweets.
            * **email:** Text embedding specialized for English emails.
            * **product-reviews:** Text embedding specialized for English product reviews.

        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param batch_size: How many instances to send to the server at a time.
        :type batch_size: int
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool
        :param timeout: HTTP timeout for request.
        :type timeout: int
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
        return self.embed(url, data, batch_size=batch_size, opts=opts, timeout=timeout)

    def embed_sentence(self, sentence, model='english', version='default',
                       opts={}, timeout=5):
        """Generate the embedding for a sentence.

        :param sentence: The sentence to embed.
        :type sentence: str
        :param model: What model to use (i.e. the kind of sentence being embedded).

            * **generic:** Generic English text embedding (the default.)
            * **reddit:** Text embedding specialized for English Reddit posts.
            * **twitter:** Text embedding specialized for English tweets.
            * **email:** Text embedding specialized for English emails.
            * **product-reviews:** Text embedding specialized for English product reviews.

        :type model: str
        :param version: What version of that model to use.
        :type version: str
        :param opts: Options specific to the model/version you chose.
        :type opts: Dict[str, Any]
        :param opts["dimensions"]: Number of dimensions to return.  PCA will be used to reduce the number of dimensions with minimal information loss.
        :type opts["dimensions"]: int
        :param opts["normalize_l2"]: Whether or not each instance should be scaled to have unit L2 norm.  (This is sometimes useful for instance retrieval tasks.)  Defaults to False.
        :type opts["normalize_l2"]: bool
        :param opts["normalize_mean"]: Whether or not to normalize each feature in the embedding to have mean 0 across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_mean"]: bool
        :param opts["normalize_variance"]: Whether or not to normalize each feature in the embedding to have unit variance across our sample dataset.  Defaults to True when `dimensions` is set, or False otherwise.
        :type opts["normalize_variance"]: bool
        :param timeout: HTTP timeout for request.
        :type timeout: int
        :returns: An embedding.
        :rtype: List[float]

        >>> with basilica.Connection('SLOW_DEMO_KEY') as c:
        ...   print(c.embed_sentence('This is a sentence.')
        [0.6246702671051025, ...]
        """
        return list(self.embed_sentences([sentence], model=model, version=version,
                                         opts=opts, timeout=timeout))[0]

    def __encode_image(self, image, transform_image):
        if type(image) != bytes:
            raise TypeError('`image` argument must be bytes (got `%s`)' % (type(image).__name__))
        if transform_image:
            try:
                im = Image.open(io.BytesIO(image))
            except IOError as e:
                raise TypeError('`image` argument must be an image (`%s`)' % (str(e)))
            except OSError as e:
                raise TypeError('`image` argument must be an image (`%s`)' % (str(e)))
            im.thumbnail((512, 512))
            im = im.convert("RGB")
            img_bytes = io.BytesIO()
            im.save(img_bytes, "JPEG")
            image = img_bytes.getvalue()
        return base64.b64encode(image).decode('utf-8')
