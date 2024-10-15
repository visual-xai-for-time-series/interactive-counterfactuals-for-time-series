import { Injectable } from '@angular/core'
import { BehaviorSubject } from 'rxjs'

@Injectable({
    providedIn: 'root',
})
export class InteractionsService {
    private highlightData = new BehaviorSubject(0)
    getHighlightData = this.highlightData.asObservable()

    private lineData = new BehaviorSubject<any[] | null>(null)
    getLineData = this.lineData.asObservable()

    private scatterData = new BehaviorSubject(null)
    getScatterData = this.scatterData.asObservable()

    private colorMode = new BehaviorSubject(2)
    getColorMode = this.colorMode.asObservable()

    private reloadData = new BehaviorSubject('')
    getReloadData = this.reloadData.asObservable()

    private data

    constructor() {}

    setColorMode(mode: number): void {
        this.colorMode.next(mode)
    }

    setHighlightData(data: any): void {
        this.highlightData.next(data)
    }

    setReloadData(baseModel: string): void {
        this.reloadData.next(baseModel)
    }

    addLineData(idx: any): void {
        let tmp = [idx, this.data[0][idx], this.data[1][idx], this.data[2][idx]]
        console.log(tmp)

        this.lineData.next(tmp)
    }

    addNewLineData(data: any): void {
        this.lineData.next(data)
    }

    addNewScatterData(data: any): void {
        this.scatterData.next(data)
    }

    clearScatterData(): void {
        this.scatterData.next(null)
    }

    getData(): any {
        return this.data
    }

    setData(data: any): void {
        console.log(data)

        this.data = data
    }
}
