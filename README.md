## Running the application

### Running full application with Docker Compose

```shell
git clone --branch guestaccess_patch --single-branch https://github.com/HU-DIPO/DecisionMiningFuzzy.git
cd DecisionMiningFuzzy
docker compose up
```

### running clean, deleting all possible legacy caches after downloading the git:

```shell
docker compose down --rmi local --remove-orphans
docker compose build --no-cache
docker compose up
```

And then go to `localhost:4200` for the frontend and `localhost:5001` for the backend.

The login screen has a button "continue as guest" to bypass the login screen.

The pseudocode of the algorithm can be found in this repository.
