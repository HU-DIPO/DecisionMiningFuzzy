import { Component, OnInit } from '@angular/core';
import { environment } from 'src/environments/environment';
@Component({
  selector: 'app-home-page',
  templateUrl: './home-page.component.html',
  styleUrls: ['./home-page.component.css']
})
export class HomePageComponent implements OnInit {

  constructor() { }

  ngOnInit() {
    this.getModels().then(data => {
      for (let key in data.models) {
        let model_dict = data.models[key];
        
        const div_model = document.createElement('div')
        const p = document.createElement('p')
        p.innerText = `${model_dict.name}:\n${model_dict.description}`
        // p.style.cssText = ''
        const div = document.getElementById('models')
        div_model.appendChild(p)
        div.appendChild(p)
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
}






		