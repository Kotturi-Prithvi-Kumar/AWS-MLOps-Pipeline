from unittest.mock import patch

from src.s3_pipeline import create_s3_client


def test_s3_client_configuration():
    with patch("src.s3_pipeline.boto3.Session") as mock_session:
        create_s3_client(
            profile="mlops-dev",
            region="ap-south-1",
        )

        mock_session.assert_called_once_with(
            profile_name="mlops-dev",
            region_name="ap-south-1",
        )

        _, kwargs = mock_session.return_value.client.call_args

        config = kwargs["config"]

        assert config.retries["mode"] == "standard"
        assert config.retries["total_max_attempts"] == 3
        assert config.connect_timeout == 5
        assert config.read_timeout == 30