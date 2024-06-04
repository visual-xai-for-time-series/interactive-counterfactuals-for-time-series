import { Component, OnInit } from '@angular/core'

import { CommonModule } from '@angular/common'

import { MatDialogContent } from '@angular/material/dialog'
import { MatDialogRef } from '@angular/material/dialog'
import { MatDialogTitle } from '@angular/material/dialog'
import { MatGridListModule } from '@angular/material/grid-list'

import { HttpService } from '../visualization/http.service'

import { environment } from '../../environments/environment'

@Component({
    selector: 'app-entry-page',
    standalone: true,
    imports: [MatDialogTitle, MatDialogContent, MatGridListModule, CommonModule],
    templateUrl: './entry-page.component.html',
    styleUrl: './entry-page.component.scss',
})
export class EntryPageComponent implements OnInit {
    private baseUrl = environment.apiURL

    public getModelsUrl = this.baseUrl + '/api/get_models'

    constructor(public dialogRef: MatDialogRef<EntryPageComponent>, private httpService: HttpService) {}

    selectOption(option: string) {
        this.dialogRef.close(option)
    }

    ngOnInit(): void {
        console.log(this.baseUrl)

        this.httpService.get<any>(this.getModelsUrl).subscribe((data: any) => {
            console.log(data)
        })
    }
}
