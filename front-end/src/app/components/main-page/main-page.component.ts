import { Component, OnInit } from '@angular/core';
import { AuthService } from '../../shared/services/auth.service';
import { environment } from 'src/environments/environment';


@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.component.html',
  styleUrls: ['./main-page.component.css']
})
export class MainPageComponent implements OnInit {

  constructor(public authService: AuthService) {
  }

  ngOnInit(): void {
    sessionStorage.clear();
    this.getModels().then(data => {
      for (let key in data.models) {
        let model_dict = data.models[key];
        
        const div_model = document.createElement('div')
        const p = document.createElement('p')
        p.innerText = `${model_dict.name}:\n${model_dict.description}`
        // p.style.cssText = ''
        const div = document.getElementById('myPopup')
        div_model.appendChild(p)
        div.appendChild(div_model)
        
        const a = document.createElement('option')
        a.innerText = `${model_dict.name}`
        a.setAttribute("value", `${model_dict.id}`)
        const select_algorithms = document.getElementById('algorithms')
        select_algorithms.appendChild(a)
    }})
  }

  async getModels() {
    const response = await fetch(environment.apiUri + "models", {
      method: 'GET',
      mode: 'cors', 
      headers: {}
    });

    return response.json()
  }

  // Retrieves the values from the selected algorithm and checks whether a file has been selected
  thisFunction() {
    const algorithm = (document.getElementById('algorithms') as HTMLInputElement).value;
    const normalize = (document.getElementById('checkbox-normalize') as HTMLInputElement).checked.toString();
    const value1 = (document.getElementById('myFile') as HTMLInputElement).value;
    if (value1 === ''){
      alert('You must select a file');
      window.location.reload();
    }
    sessionStorage.setItem('algorithms', algorithm);
    sessionStorage.setItem('normalize_bool', normalize);
  }
  // Functionality for question mark pop-up
  myPopup() {
    const popup = document.getElementById('myPopup');
    popup.classList.toggle('show');
  }
}
