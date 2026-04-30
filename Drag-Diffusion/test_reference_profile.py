from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from eval.reference_profile import ReferenceProfile, score_image_against_reference
from eval_reference import resolve_image


def make_scene(
    *,
    grass_color=(52, 146, 34),
    dog_box=(205, 300, 290, 355),
    dog_color=(238, 226, 196),
    size=(512, 512),
):
    image = Image.new("RGB", size, grass_color)
    draw = ImageDraw.Draw(image)
    if dog_box is not None:
        draw.ellipse(dog_box, fill=dog_color)
    return image


def test_reference_like_scene_scores_higher_than_bad_scene():
    reference_like = make_scene()
    bad_scene = make_scene(grass_color=(120, 120, 120), dog_box=None)

    reference_score = score_image_against_reference(reference_like)
    bad_score = score_image_against_reference(bad_scene)

    assert reference_score.overall >= 0.80
    assert bad_score.overall < reference_score.overall
    assert bad_score.dog_presence == 0.0


def test_dog_position_penalizes_far_from_reference_location():
    centered = make_scene()
    far_right = make_scene(dog_box=(385, 300, 470, 355))

    centered_score = score_image_against_reference(centered)
    far_right_score = score_image_against_reference(far_right)

    assert centered_score.dog_position > far_right_score.dog_position


def test_custom_profile_accepts_different_expected_position():
    far_right = make_scene(dog_box=(385, 300, 470, 355))
    profile = ReferenceProfile(dog_center_x=(0.72, 0.95), dog_center_y=(0.55, 0.75))

    score = score_image_against_reference(far_right, profile)

    assert score.dog_position >= 0.90


def test_latest_result_ignores_reference_report():
    with TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        result = result_dir / "candidate_ours.png"
        report = result_dir / "reference_eval_report.png"
        Image.new("RGB", (8, 8), "green").save(result)
        Image.new("RGB", (8, 8), "black").save(report)

        assert resolve_image("latest", result_dir) == result


def test_latest_result_prefers_ours_over_baseline():
    with TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        baseline = result_dir / "candidate_baseline.png"
        ours = result_dir / "candidate_ours.png"
        Image.new("RGB", (8, 8), "blue").save(ours)
        Image.new("RGB", (8, 8), "red").save(baseline)

        assert resolve_image("latest", result_dir) == ours
