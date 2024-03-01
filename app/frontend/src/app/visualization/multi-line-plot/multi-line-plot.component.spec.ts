import { ComponentFixture, TestBed } from '@angular/core/testing'

import { LinePlotComponent } from './multi-line-plot.component'

describe('LinePlotComponent', () => {
    let component: LinePlotComponent
    let fixture: ComponentFixture<LinePlotComponent>

    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [LinePlotComponent],
        }).compileComponents()

        fixture = TestBed.createComponent(LinePlotComponent)
        component = fixture.componentInstance
        fixture.detectChanges()
    })

    it('should create', () => {
        expect(component).toBeTruthy()
    })
})
