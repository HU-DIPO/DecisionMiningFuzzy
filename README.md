## Running the application

### Running full application with Docker Compose

```shell
git clone --branch guestaccess_patch --single-branch https://github.com/HU-DIPO/DecisionMiningFuzzy.git
cd DecisionMiningFuzzy
docker compose up
```

And then go to `localhost:4200` for the frontend and `localhost:5001` for the backend.

The login screen has a button "continue as guest" to bypass the login screen.

The pseudocode of the algorithm can be found in this repository.
