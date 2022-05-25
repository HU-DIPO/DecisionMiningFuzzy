import { NgModule } from '@angular/core';
// Required services for navigation
import { Routes, RouterModule } from '@angular/router';

// Import all the components for which navigation service has to be activated
import { SignInComponent } from './components/sign-in/sign-in.component';
import { SignUpComponent } from './components/sign-up/sign-up.component';
import { ForgotPasswordComponent } from './components/forgot-password/forgot-password.component';
import { VerifyEmailComponent } from './components/verify-email/verify-email.component';
import { AuthGuard} from './shared/guard/auth.guard';
import { MainPageComponent } from './components/main-page/main-page.component';
import { DmnPageComponent } from './components/dmn-page/dmn-page.component';
import {ProfilePageComponent} from './components/profile-page/profile-page.component';
import {SecondPageComponent} from './components/second-page/second-page.component';
import {HomePageComponent} from './components/home-page/home-page.component';


const routes: Routes = [
  { path: '', redirectTo: '/sign-in', pathMatch: 'full' },
  { path: 'sign-in', component: SignInComponent },
  { path: 'register-user', component: SignUpComponent },
  { path: 'forgot-password', component: ForgotPasswordComponent },
  { path: 'verify-email-address', component: VerifyEmailComponent },
  { path: 'home-page', component: HomePageComponent, canActivate: [AuthGuard] },
  { path: 'profile-page', component: ProfilePageComponent, canActivate: [AuthGuard] },
  { path: 'dmn-page', component: DmnPageComponent, canActivate: [AuthGuard] },
  { path: 'main-page', component: MainPageComponent, canActivate: [AuthGuard] },
  { path: 'second-page', component: SecondPageComponent, canActivate: [AuthGuard]}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})

export class AppRoutingModule { }
