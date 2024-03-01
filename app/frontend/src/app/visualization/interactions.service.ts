import { Injectable } from '@angular/core'
import { BehaviorSubject } from 'rxjs'

@Injectable({
    providedIn: 'root',
})
export class InteractionsService {
    private highlightData = new BehaviorSubject(0)
    getHighlightData = this.highlightData.asObservable()

    private lineData = new BehaviorSubject([0, 0, 0, 0])
    getLineData = this.lineData.asObservable()

    private scatterData = new BehaviorSubject(null)
    getScatterData = this.scatterData.asObservable()

    private data

    constructor() {}

    setHighlightData(data: any): void {
        this.highlightData.next(data)
    }

    addLineData(idx: any): void {
        let tmp = [idx, this.data[0][idx], this.data[1][idx], this.data[2][idx]]
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
        this.data = data
    }
}
