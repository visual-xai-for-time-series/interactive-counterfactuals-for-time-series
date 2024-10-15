import { Component, OnInit } from '@angular/core'

import { CommonModule } from '@angular/common'
import { MatGridListModule } from '@angular/material/grid-list'
import { MatTabsModule } from '@angular/material/tabs'
import { MatProgressBarModule } from '@angular/material/progress-bar'

import { MatIconModule } from '@angular/material/icon'
import { MatDividerModule } from '@angular/material/divider'
import { MatButtonModule } from '@angular/material/button'

import { ScatterPlotComponent } from './scatter-plot/scatter-plot.component'
import { LinePlotsComponent } from './line-plots/line-plots.component'

import { InteractionsService } from './interactions.service'
import { HttpService } from './http.service'
import { DataService } from './data.service'

import { environment } from '../../environments/environment'

@Component({
    selector: 'app-visualization',
    standalone: true,
    imports: [
        ScatterPlotComponent,
        LinePlotsComponent,
        CommonModule,
        MatGridListModule,
        MatTabsModule,
        MatProgressBarModule,
        MatIconModule,
        MatDividerModule,
        MatButtonModule,
    ],
    templateUrl: './visualization.component.html',
    styleUrl: './visualization.component.scss',
})
export class VisualizationComponent implements OnInit {
    private dataset = 'resnet-wafer' // resnet-ecg5000 resnet-wafer resnet-forda

    private stage = 'train'

    public isRequestPending = true

    private baseUrl = environment.apiURL

    public projectedTimeSeries = [
        'Projected Time Series',
        this.baseUrl + '/api/get_projected_time_series/?stage=' + this.stage,
        this.baseUrl + '/api/get_projected_time_series_density/?stage=' + this.stage,
        this.baseUrl + '/api/project_time_series/?stage=' + this.stage,
        this.baseUrl + '/api/inverse_project_time_series/?stage=' + this.stage,
    ]
    public projectedActivations = [
        'Projected Activations',
        this.baseUrl + '/api/get_projected_activations/?stage=' + this.stage,
        this.baseUrl + '/api/get_projected_activations_density/?stage=' + this.stage,
        this.baseUrl + '/api/project_activations/?stage=' + this.stage,
        this.baseUrl + '/api/inverse_project_activations/?stage=' + this.stage,
    ]
    public projectedAttributions = [
        'Projected Attributions',
        this.baseUrl + '/api/get_projected_attributions/?stage=' + this.stage,
        this.baseUrl + '/api/get_projected_attributions_density/?stage=' + this.stage,
        this.baseUrl + '/api/project_attributions/?stage=' + this.stage,
        this.baseUrl + '/api/inverse_project_attributions/?stage=' + this.stage,
    ]

    public colorScales = this.baseUrl + '/api/get_color_scale/'

    public originalTimeSeries = this.baseUrl + '/api/get_time_series/?stage=' + this.stage

    public setBaseModel = this.baseUrl + '/api/set_model/?model='

    public firstDataLoaded = false

    constructor(
        private httpService: HttpService,
        private interactionsService: InteractionsService,
        private dataService: DataService
    ) {}

    loadData(baseModel): void {
        this.httpService.post<any>(this.setBaseModel + baseModel, {}).subscribe((data: any) => {
            console.log(data)

            console.log('Model Loaded')

            this.httpService.get<any>(this.colorScales).subscribe((data: any) => {
                const parsed_data = JSON.parse(data)
                this.dataService.setColorScale(parsed_data['color'])

                console.log('Color Scale Loaded')

                this.httpService.get<any>(this.originalTimeSeries).subscribe((data: any) => {
                    const parsed_data = JSON.parse(data)
                    this.interactionsService.setData(parsed_data['data'])

                    this.firstDataLoaded = true
                    console.log('Data Loaded')
                })
            })
        })
    }

    ngOnInit(): void {
        console.log(this.baseUrl)
        this.firstDataLoaded = false

        this.loadData(this.dataset)

        this.httpService.isRequestPending().subscribe((pending: boolean) => {
            this.isRequestPending = pending
        })

        this.interactionsService.getReloadData.subscribe((data) => {
            if (data != '') {
                this.loadData(data)
            }
        })
    }

    setColorMode(mode: number = 2) {
        this.interactionsService.setColorMode(mode)
    }
}
