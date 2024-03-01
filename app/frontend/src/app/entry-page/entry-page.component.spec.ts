import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EntryPageComponent } from './entry-page.component';

describe('EntryPageComponent', () => {
  let component: EntryPageComponent;
  let fixture: ComponentFixture<EntryPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EntryPageComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EntryPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
