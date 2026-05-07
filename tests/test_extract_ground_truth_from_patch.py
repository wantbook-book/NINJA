from __future__ import annotations

import unittest

from examples.data_preprocess.extract_ground_truth_from_patch import extract_edit_functions_from_patch


class ExtractGroundTruthFromPatchTest(unittest.TestCase):
    def test_nested_function_in_class_method_collapses_to_outer_method(self) -> None:
        patch = """diff --git a/astropy/wcs/wcs.py b/astropy/wcs/wcs.py
--- a/astropy/wcs/wcs.py
+++ b/astropy/wcs/wcs.py
@@ -1,7 +1,7 @@
 class WCS:
     def _array_converter(self):
         def _return_list_of_arrays():
-            return []
+            return [value]
         def _return_single_array():
             return None
"""

        self.assertEqual(
            extract_edit_functions_from_patch(patch),
            ["astropy/wcs/wcs.py:WCS._array_converter"],
        )

    def test_nested_function_in_top_level_function_collapses_to_outer_function(self) -> None:
        patch = """diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -1,6 +1,6 @@
 def generate():
     def iter_content():
-        return b""
+        return b"x"
     return iter_content()
"""

        self.assertEqual(
            extract_edit_functions_from_patch(patch),
            ["requests/models.py:generate"],
        )

    def test_direct_class_method_is_kept(self) -> None:
        patch = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,5 +1,5 @@
 class Foo:
     def bar(self):
-        return 1
+        return 2
"""

        self.assertEqual(
            extract_edit_functions_from_patch(patch),
            ["pkg/mod.py:Foo.bar"],
        )

    def test_nested_function_in_top_level_function_collapses_to_outer_function_2(self) -> None:
        patch = """iff --git a/astropy/wcs/wcs.py b/astropy/wcs/wcs.py
--- a/astropy/wcs/wcs.py
+++ b/astropy/wcs/wcs.py
@@ -1212,6 +1212,9 @@ def _array_converter(self, func, sky, *args, ra_dec_order=False):
         def _return_list_of_arrays(axes, origin):
+            if any([x.size == 0 for x in axes]):
+                return axes
+
             try:
                 axes = np.broadcast_arrays(*axes)
             except ValueError:
@@ -1235,6 +1238,8 @@ def _return_single_array(xy, origin):
                 raise ValueError(
                     "When providing two arguments, the array must be "
                     "of shape (N, {0})".format(self.naxis))
+            if 0 in xy.shape:
+                return xy
             if ra_dec_order and sky == 'input':
                 xy = self._denormalize_sky(xy)
             result = func(xy, origin)
"""

        self.assertEqual(
            extract_edit_functions_from_patch(patch),
            ["astropy/wcs/wcs.py:WCS._array_converter"],
        )


if __name__ == "__main__":
    unittest.main()
