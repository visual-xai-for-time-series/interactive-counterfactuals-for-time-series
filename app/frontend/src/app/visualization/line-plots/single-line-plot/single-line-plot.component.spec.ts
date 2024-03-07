import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SingleLinePlotComponent } from './single-line-plot.component';

describe('SingleLinePlotComponent', () => {
  let component: SingleLinePlotComponent;
  let fixture: ComponentFixture<SingleLinePlotComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SingleLinePlotComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SingleLinePlotComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
