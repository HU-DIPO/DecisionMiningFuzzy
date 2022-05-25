import { Component, OnInit } from '@angular/core';
import { AuthService } from '../../shared/services/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-sign-in',
  templateUrl: './sign-in.component.html',
  styleUrls: ['./sign-in.component.css']
})
export class SignInComponent implements OnInit {

  constructor(public authService: AuthService, router: Router) {
    if (authService.isLoggedIn) {
      router.navigate(['home-page']);
    }
  }

  ngOnInit(): void {
  }

}
