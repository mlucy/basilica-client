===============
Basilica
===============

Client bindings for basilica.ai.

Supported Python versions: 2.7

Releases
========
- 0.2.1 : 10-15-2018
  - Bug fix.
- 0.2.0 : 10-15-2018
  - Added docstrings, minor API change.
- 0.1.1 : 10-12-2018
  - Fixing description.
- 0.1.0 : 10-12-2018
  - Initial release.

Documentation
=============
Full documentation is available online at http://basilica.ai/docs .

Installation
============
1. Install the bindings using pip

    pip install basilica

3. Sign up at http://basilica.ai/signup to get an API key.

Connecting to Basilica
===========
You can connect to Basilica using your API key::

    import basilica

    with basilica.Connection(API_KEY) as c:
      ...

If you don't have an API key, you can visit http://basilica.ai/signup
to get one.

Embedding Images
===========

You can embed a single image with `Connection.embed_image`, which
takes bytes, or `Connection.embed_image_file`, which takes a path to a
file.  There are also helpers `Connection.embed_images` and
`Connction.embed_image_files`, which will return a generator that
batches requests to the server::

    import basilica

    with basilica.Connection(API_KEY) as c:
        image_embedding = c.embed_image(BYTES)
        image_embedding = c.embed_image_file('/path/to/image/')
        for image_embedding in c.embed_images([BYTES1, BYTES2, ...]):
            ...
        for image_embedding in c.embed_image_files(['/path1', '/path2', ...]):
            ...

Embedding Sentences
===========

You can embed a single sentence with `Connection.embed_sentence`,
which takes a utf-8 string, or with `Connection.embed_sentences`,
which will return a generator that batches requests to the server::

    import basilica

    with basilica.Connection(API_KEY) as c:
        sentence_embedding = c.embed_sentence(BYTES)
        for sentence_embedding in c.embed_sentences([BYTES1, BYTES2, ...]):
            ...
