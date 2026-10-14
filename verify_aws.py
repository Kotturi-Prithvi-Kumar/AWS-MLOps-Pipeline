import boto3

profile = os.getenv("AWS_PROFILE", "mlops-dev")
region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

session = boto3.Session(
    profile_name=profile,
    region_name=region,
)

sts = session.client("sts")

identity = sts.get_caller_identity()

print("AWS authentication successful!")
print("Account:", identity["Account"])
print("Region:", session.region_name)
print("Role:", identity["Arn"].split("/")[-2])
