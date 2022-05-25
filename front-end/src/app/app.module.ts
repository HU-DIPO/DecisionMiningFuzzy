import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { AngularFireModule } from '@angular/fire';
import { AngularFireAuthModule } from '@angular/fire/auth';
import { AngularFirestoreModule } from '@angular/fire/firestore';
import { environment } from '../environments/environment';
import { SignInComponent } from './components/sign-in/sign-in.component';
import { SignUpComponent } from './components/sign-up/sign-up.component';
import { ForgotPasswordComponent } from './components/forgot-password/forgot-password.component';
import { VerifyEmailComponent } from './components/verify-email/verify-email.component';
import { AppRoutingModule } from './app-routing.module';
import { AuthService } from './shared/services/auth.service';
import { MainPageComponent } from './components/main-page/main-page.component';
import { DmnPageComponent } from './components/dmn-page/dmn-page.component';
import { HttpClientModule } from '@angular/common/http';
import { ProfilePageComponent } from './components/profile-page/profile-page.component';
import { SecondPageComponent } from './components/second-page/second-page.component';
import { HomePageComponent } from './components/home-page/home-page.component';

@NgModule({
  declarations: [
    AppComponent,
    SignInComponent,
    SignUpComponent,
    ForgotPasswordComponent,
    VerifyEmailComponent,
    MainPageComponent,
    DmnPageComponent,
    ProfilePageComponent,
    SecondPageComponent,
    HomePageComponent,
  ],
  imports: [
    BrowserModule,
    AngularFireModule.initializeApp(environment.firebase),
    AngularFireAuthModule,
    AngularFirestoreModule,
    AppRoutingModule,
    HttpClientModule,
  ],
  providers: [AuthService, HttpClientModule],
  bootstrap: [AppComponent]
})
export class AppModule { }
