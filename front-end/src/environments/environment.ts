// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  apiUri: 'http://localhost:5000/',
  firebase: {
    apiKey: 'AIzaSyAouDyy92iKkzgDrlZ-7rF6TsZNQ7H5RAU',
    authDomain: 'decision-mining.firebaseapp.com',
    projectId: 'decision-mining',
    storageBucket: 'decision-mining.appspot.com',
    messagingSenderId: '549838137533',
    appId: '1:549838137533:web:22976120dcff6ae71eefd4',
    measurementId: 'G-165744VC0E'
  }
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.
