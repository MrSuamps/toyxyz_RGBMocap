import unittest

from mocap.cli.placement_check import contact_moved, summary_lines

RIGHT_ARM_CONTACT = [{'joints': [14, 17, 19, 21], 'start': 243, 'end': 270}]


class ContactMovedTest(unittest.TestCase):
    def test_contact_on_the_placed_chain_inside_the_row(self):
        self.assertTrue(contact_moved(RIGHT_ARM_CONTACT, 'RightArm', 247, 266))

    def test_other_chain_or_frames_do_not_count(self):
        self.assertFalse(contact_moved(RIGHT_ARM_CONTACT, 'LeftArm', 247, 266))
        self.assertFalse(contact_moved(RIGHT_ARM_CONTACT, 'RightArm', 270, 300))
        self.assertFalse(contact_moved(None, 'RightArm', 247, 266))

    def test_summary_names_the_contact(self):
        stats = dict(median=0.1, p90=0.2)
        row = dict(id='ra-place', chain='RightArm', frames=[247, 265], before=stats, solved=stats, final=stats)
        moved = summary_lines(dict(intervals=[dict(row, kept=False, moved_by_contact=True)]))[-1]
        undone = summary_lines(dict(intervals=[dict(row, kept=False, moved_by_contact=False)]))[-1]
        self.assertIn('moved by a self-contact', moved)
        self.assertIn('NO: a later pass', undone)


if __name__ == '__main__':
    unittest.main()
