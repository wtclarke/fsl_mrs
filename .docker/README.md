## FSL-MRS Docker Images
Docker files to create Docker images for testing FSL-MRS

The iamges are available on [Docker Hub](https://hub.docker.com/u/wtclarke).

To (manually) build these images run these steps, incrementing the `{tag}` value as appropriate:

```
cd .docker/fsl_mrs_tests
docker build --platform linux/amd64 -t wtclarke/fsl_mrs_tests:{tag} -t wtclarke/fsl_mrs_tests:latest -f Dockerfile ..
docker login
docker push wtclarke/fsl_mrs_tests:{tag}
docker push wtclarke/fsl_mrs_tests:latest
```