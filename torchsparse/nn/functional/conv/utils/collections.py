# Adapted from from python-attributedict
# https://github.com/grimen/python-attributedict/blob/master/attributedict/collections.py

# Copyright (c) 2018 Jonas Grimfelt <grimen@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================
#       IMPORT
# --------------------------------------

import rootpath

import collections

from . import compat

# =========================================
#       CONSTANTS
# --------------------------------------

DEFAULT_RESERVED_KEY_PREFIX = "__"
DEFAULT_RESERVED_KEY_SUFFIX = None


# =========================================
#       CLASSES
# --------------------------------------


class AttributeDict(dict):

    """
    :class:`~attributedict.collections.AttributeDict` is a seamlessly extended dictionary object (subclass of `dict`),
    with access to additional attribute get/set/delete of key/values.

    @example:

        data = AttributeDict({'foo': {'bar': [1, 2, 3]}})

        data.foo # => `{'bar': [1, 2, 3]}}`
        data.foo.bar # => `[1, 2, 3]`

        data.foo = {'baz': True}
        data.foo = # => `{'baz': True}`

        del data.foo

    """

    def __init__(self, entries={}):
        entries = entries or {}
        entries = self._reject_reserved_keys(entries)

        super(AttributeDict, self).__init__(entries)

        self.update(entries)

    def _refresh(self):
        # HACK
        #
        # to make object encoding (e.g. `json` work), call `dict` constructor with updated data.
        #
        # it is terrible language design that this is the only way. >:/
        #
        # @see https://stackoverflow.com/questions/23088565/make-a-custom-class-json-serializable
        # @see https://stackoverflow.com/questions/2144988/python-multiple-calls-to-init-on-the-same-instance
        #
        # print('_refresh')
        dict.__init__(self, self.__dict__)

    def _reject_reserved_keys(self, object={}):
        # NOTE:
        # tricky to override on the instance, because only special attribute `self__dict__` is not causing
        # recursive calls to `self.__getattr__`. =S
        reserved_key_prefix = DEFAULT_RESERVED_KEY_PREFIX
        reserved_key_suffix = DEFAULT_RESERVED_KEY_SUFFIX

        if isinstance(object, dict):
            for key, value in list(object.items()):
                is_reserved = False

                if len(reserved_key_prefix or ""):
                    is_reserved = key.startswith(reserved_key_prefix)

                if len(reserved_key_suffix or ""):
                    is_reserved = is_reserved or key.startswith(reserved_key_suffix)

                if is_reserved:
                    del object[key]

                else:
                    object[key] = self._reject_reserved_keys(object[key])

        return object

    def update(self, entries={}, *args, **kwargs):
        """
        Update dictionary.

        @example:

            object.update({'foo': {'bar': 1}})

        """
        if isinstance(entries, dict):
            entries = self._reject_reserved_keys(entries)

        for key, value in dict(entries, *args, **kwargs).items():
            if isinstance(value, dict):
                self.__dict__[key] = AttributeDict(value)
            else:
                self.__dict__[key] = value

        self._refresh()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self, *args, **kwargs):
        return self.__dict__.items(*args, **kwargs)

    def iterkeys(self, *args, **kwargs):
        self.__dict__.iterkeys(*args, **kwargs)

    def itervalues(self, *args, **kwargs):
        self.__dict__.itervalues(*args, **kwargs)

    def iteritems(self, *args, **kwargs):
        self.__dict__.items(*args, **kwargs)

    def get(self, key, default=None):
        result = self.__dict__.get(key, default)

        return result

    def pop(self, key, value=None):
        result = self.__dict__.pop(key, value)

        self._refresh()

        return result

    def copy(self):
        return type(self)(self)

    def setdefault(self, key, default=None):
        result = self.__dict__.setdefault(key, default)

        self._refresh()

        return result

    def __getitem__(self, key):
        """
        Provides `dict` style property access to dictionary key-values.

        @example:

            value = object['key']

            # ignored
            object['__key__']

        """
        result = self.__dict__.__getitem__(key)

        self._refresh()

        return result

    def __setitem__(self, key, value):
        """
        Provides `dict` style property assignment to dictionary key-values.

        @example:

            object['key'] = value

            # ignored
            object['__key__'] = value

        """
        if isinstance(value, dict):
            value = AttributeDict(value)

        result = self.__dict__.__setitem__(key, value)

        self._refresh()

        return result

    def __delitem__(self, key):
        """
        Provides `dict` style property deletion to dictionary key-values.

        @example:

            del object['key']

            # ignored
            del object['__key__']

        """
        result = self.__dict__.__delitem__(key)

        self._refresh()

        return result

    def __getattr__(self, key):
        """
        Provides `object` style attribute access to dictionary key-values.

        @example:

            value = object.key

            # ignored
            object.__key__

        """
        try:
            return self.__getitem__(key)

        except Exception as error:
            raise AttributeError(error)

    def __setattr__(self, key, value):
        """
        Provides `object` style attribute assignment to dictionary key-values.

        @example:

            object.key = value

            # ignored
            object.__key__ = value

        """
        try:
            return self.__setitem__(key, value)

        except Exception as error:
            raise AttributeError(error)

    def __delattr__(self, key):
        """
        Provides `object` style attribute deletion to dictionary key-values.

        @example:

            del object.key

            # ignored
            del object.__key__

        """
        try:
            return self.__delitem__(key)

        except Exception as error:
            raise AttributeError(error)

    def __str__(self):
        """
        String value of the dictionary instance.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        String representation of the dictionary instance.
        """
        return repr(self.__dict__)

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__.keys())

    def __iter__(self):
        """
        Iterate over dictionary key/values.
        """
        return iter(self.__dict__.keys())

    def __len__(self):
        """
        Get number of items.
        """
        return len(self.__dict__.keys())

    def __contains__(self, key):
        """
        Check if key exists.
        """
        return self.__dict__.__contains__(key)

    def __reduce__(self):
        """
        Return state information for pickling.
        """
        return self.__dict__.__reduce__()

    def __eq__(self, other):
        """
        Check dictionary is equal to another provided dictionary.
        """
        return self.__dict__.__eq__(other)

    def __ne__(self, other):
        """
        Check dictionary is inequal to another provided dictionary.
        """
        return self.__dict__.__ne__(other)

    def to_dict(self):
        return self.__class__.dict(self.__dict__)

    @classmethod
    def fromkeys(klass, keys, value=None):
        return AttributeDict(dict.fromkeys((key for key in keys), value))

    @classmethod
    def dict(klass, _dict={}):
        if _dict is None:
            return None

        new_dict = {}

        for key, value in _dict.items():
            if isinstance(value, AttributeDict):
                new_dict[key] = AttributeDict.to_dict(value)
            else:
                new_dict[key] = value

        return new_dict


# =========================================
#       EXPORT
# --------------------------------------

attributedict = AttributeDict
attrdict = AttributeDict

__all__ = [
    "AttributeDict",
    "attributedict",
    "attrdict",
]


# =========================================
#       MAIN
# --------------------------------------

if __name__ == "__main__":

    data = {"a": {"b": {"c": [1, 2, 3]}}}

    object = AttributeDict(data)

    print("object = AttributeDict({0})\n".format(data))

    print("object\n\n\t{0}\n".format(object))
    print("object.a\n\n\t{0}\n".format(object.a))
    print("object.a.b\n\n\t{0}\n".format(object.a.b))
    print("object.a.b.c\n\n\t{0}\n".format(object.a.b.c))
