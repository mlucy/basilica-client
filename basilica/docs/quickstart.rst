.. _quickstart:

Quickstart
==========

Install the python client
^^^^^^^^^^^^^^^^^^^^^^^^^

First, install the Python client.

.. code-block:: bash

   $ pip install basilica

Embed some sentences
^^^^^^^^^^^^^^^^^^^^

Let's embed some sentences to make sure the client is working.

.. code-block:: python

   import basilica
   sentences = [
       "This is a sentence!",
       "This is a similar sentence!",
       "I don't think this sentence is very similar at all...",
   ]
   with basilica.Connection('SLOW_DEMO_KEY') as c:
       embeddings = list(c.embed_sentences(sentences))
   print(embeddings)

::

   [[0.8556405305862427, ...], ...]

Let's also make sure these embeddings make sense, by checking that the
cosine distance between the two similar sentences is smaller:

.. code-block:: python

   from scipy import spatial
   print(spatial.distance.cosine(embeddings[0], embeddings[1]))
   print(spatial.distance.cosine(embeddings[0], embeddings[2]))

::

   0.024854343247535327
   0.25084750542635814

Great!

Get an API key
^^^^^^^^^^^^^^

The example above uses the slow demo key.  You can get an API key of
your own by signing up at https://basilica.ai/signup .  (If you
already have an account, you can view your API keys at
https://basilica.ai/api_keys .)

What next?
^^^^^^^^^^

* Read the documentation for the python client: :ref:`basilica`
* Check out some demos to get inspired: https://basilica.ai/demos

