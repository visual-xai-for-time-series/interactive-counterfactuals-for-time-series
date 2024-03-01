import { Component } from '@angular/core'
import { RouterOutlet } from '@angular/router'
import { VisualizationComponent } from './visualization/visualization.component'

@Component({
    selector: 'app-root',
    standalone: true,
    imports: [RouterOutlet, VisualizationComponent],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss',
})
export class AppComponent {
    title = 'Interactive Generation of Counterfactuals for Time Series'
}
