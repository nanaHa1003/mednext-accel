from mednext_accel.optimization.profiles import load_bundled_profile


def test_bundled_profiles_parse_and_have_family_defaults() -> None:
    for name in ("generic-nvidia", "sm120"):
        profile = load_bundled_profile(name)
        assert profile.name == name
        assert "pointwise_conv3d" in profile.defaults
        assert "depthwise_conv3d" in profile.defaults
