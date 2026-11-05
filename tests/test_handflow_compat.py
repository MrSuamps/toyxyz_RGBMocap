import unittest
from unittest import mock

from mocap import handflow
from mocap.result_artifacts import file_stat
from mocap.storage import sha256


class HandFlowCodeCompatTest(unittest.TestCase):
    def setUp(self):
        self.stat = file_stat(handflow.code_path())
        self.digest = sha256(handflow.code_path())

    def test_same_stat_is_current(self):
        self.assertTrue(handflow.code_current({'code_stat': self.stat}))

    def test_reviewed_writer_size_is_current(self):
        with mock.patch.dict('mocap.cache_compat.HANDFLOW_CODE_COMPAT', {8286: self.digest}, clear=True):
            self.assertTrue(handflow.code_current({'code_stat': [8286, 1]}))
            self.assertFalse(handflow.code_current({'code_stat': [8287, 1]}))

    def test_reviewed_writer_digest_is_current(self):
        with mock.patch.dict('mocap.cache_compat.HANDFLOW_CODE_COMPAT', {'old': self.digest}, clear=True):
            self.assertTrue(handflow.code_current({'code_stat': [1, 1], 'code': 'old'}))
            self.assertFalse(handflow.code_current({'code_stat': [1, 1], 'code': 'other'}))

    def test_approval_requires_the_current_code(self):
        with mock.patch.dict('mocap.cache_compat.HANDFLOW_CODE_COMPAT', {8286: '0' * 64}, clear=True):
            self.assertFalse(handflow.code_current({'code_stat': [8286, 1]}))


if __name__ == '__main__':
    unittest.main()
