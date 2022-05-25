import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DmnPageComponent } from './dmn-page.component';

describe('DmnPageComponent', () => {
  let component: DmnPageComponent;
  let fixture: ComponentFixture<DmnPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DmnPageComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DmnPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
